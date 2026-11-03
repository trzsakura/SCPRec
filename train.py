
import os
import json
import datasets
import rich
import torch
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from transformers import (
    EarlyStoppingCallback,
    GenerationConfig,
    TrainerCallback,
    set_seed,
)
from data_collators.data_collator import RecommendationDataCollator as DataCollator
from paths import model_names
from trainers.utils import get_compute_metrics, get_tokenizer, MetricUpdater
from trainers.OutcomeGRPOTrainer import (
    OutcomeGRPOTrainer,
    OutcomeGRPOTrainingArguments)
from training_protocols import (
    checkpoint_needs_evaluation,
    last_prefixed_metrics,
    resolve_resume_checkpoint,
)


_PERIODIC_CHECKPOINT_STEPS = 500


class PeriodicCheckpointCallback(TrainerCallback):
    """Request an additional resumable checkpoint every 500 training steps."""

    def on_step_end(self, args, state, control, **kwargs):
        del args, kwargs
        if (
                state.global_step > 0
                and state.global_step % _PERIODIC_CHECKPOINT_STEPS == 0
        ):
            control.should_save = True
        return control


def train(
        output_dir="../checkpoints",

        run_name: str = "debug-v2",

        train_batch_size: int = 4,
        eval_batch_size: int = 32,

        train_generation_batch_size=16,
        test_generation_batch_size=16,

        item_emb_batch_size: int = 128,
        warmup_steps: int = 32,

        eval_freq=8,
        early_stopping_patience=8,
        eval_on_start: bool = True,
        fixed_epoch_combined_eval: bool = False,
        evaluation_only: bool = False,
        gradient_accumulation_steps: int = 1,

        num_train_epochs: int = 10,
        learning_rate: float = 1e-5,


        cleanup_previous_checkpoints=False,
        
        dataset_category: str = "CDs_and_Vinyl",
        dataset_dir="data/CDs_and_Vinyl_0_2022-10-2023-10",
        
        use_lora=True,
        init_checkpoint=None,
        seed=42,
        model = 'qwen',
        model_name_or_path=None,

        resume_from_checkpoint: bool = False,
        window_size: int = 20,
        gather_negs_across_processes=True,
        
        lr_scheduler_type='constant',
        use_vllm=True,


        max_new_tokens=256,
        group_size=4,
        gen_top_k=200,
        gen_temperature=2.,
        gen_top_p=1.0,

        **kwargs,
):
    # Trainer seeds the RNG during its own initialization, but the model and
    # fresh LoRA adapter are created before that point.  Seed here as well so
    # repeated launches start from the same adapter initialization.
    set_seed(seed)

    trainer_extra_kwargs = dict()
    lora_kwargs = dict()
    for k in kwargs:
        if k.startswith('trainer'):
            trainer_extra_kwargs[k.replace('trainer_', '')] = kwargs[k]
        else:
            lora_kwargs[k] = kwargs[k]
    del kwargs

    if evaluation_only:
        if not fixed_epoch_combined_eval:
            raise ValueError(
                "evaluation_only requires fixed_epoch_combined_eval=True."
            )
        if not init_checkpoint:
            raise ValueError(
                "evaluation_only requires init_checkpoint to identify the "
                "trained LoRA adapter."
            )
        if resume_from_checkpoint not in (False, None, "", "false", "False"):
            raise ValueError(
                "evaluation_only loads weights through init_checkpoint and "
                "does not accept resume_from_checkpoint."
            )
        resume_from_checkpoint = False

    if fixed_epoch_combined_eval:
        epoch_count = float(num_train_epochs)
        if not epoch_count.is_integer() or epoch_count <= 0:
            raise ValueError(
                "fixed_epoch_combined_eval requires num_train_epochs to be "
                "a positive integer."
            )
        if cleanup_previous_checkpoints:
            raise ValueError(
                "fixed_epoch_combined_eval keeps resumable epoch-boundary "
                "and periodic checkpoints, so cleanup_previous_checkpoints "
                "must be False."
            )
        # A user-provided trainer_max_steps must never shorten the requested
        # fixed epoch budget.  Other protocol fields are explicit below.
        trainer_extra_kwargs["max_steps"] = -1
        for forced_key in (
                "eval_strategy",
                "eval_on_start",
                "save_strategy",
                "save_total_limit",
                "load_best_model_at_end",
                "metric_for_best_model",
        ):
            trainer_extra_kwargs.pop(forced_key, None)

    datasets.disable_progress_bars()
    if model == 'gemma':
        model_name = model_name_or_path or model_names["Gemma-2-2b-it"]
        from models.gemma_models import (Gemma2RecommendationCasualLM as ModelClass,
                                         Gemma2RecommendationConfig as ConfigClass)
    elif model == 'qwen':
        model_name = model_name_or_path or model_names["Qwen2.5-3B-Instruct"]
        from models.qwen_models import (Qwen2RecommendationCasualLM as ModelClass,
                                        Qwen2RecommendationConfig as ConfigClass)
    else:
        raise NotImplementedError
    output_dir = os.path.join(output_dir, run_name)
    accelerator = Accelerator()
    fixed_resume_needs_eval = False
    if fixed_epoch_combined_eval:
        resume_from_checkpoint = resolve_resume_checkpoint(
            resume_from_checkpoint,
            run_dir=output_dir,
        )
        if not resume_from_checkpoint:
            # Every rank must inspect the same pre-creation filesystem state.
            # Without both barriers, a faster rank can create output_dir and
            # make a slower rank misclassify this fresh distributed run as an
            # existing experiment.
            accelerator.wait_for_everyone()
            output_dir_already_existed = os.path.exists(output_dir)
            accelerator.wait_for_everyone()
            if output_dir_already_existed:
                raise FileExistsError(
                    "A fresh fixed-epoch run will not reuse an existing run "
                    f"directory: {output_dir}. Change run_name or resume a "
                    "checkpoint from that run."
                )
        if resume_from_checkpoint:
            fixed_resume_needs_eval = checkpoint_needs_evaluation(
                resume_from_checkpoint
            )

    rich.print(accelerator.deepspeed_plugin)

    if accelerator.is_main_process:
        rich.print("Arguments: ", locals())

    ################## set dataset ##################

    dset = datasets.load_from_disk(dataset_dir)


    tokenizer = get_tokenizer(model_name)

    emb_token = '<answer>'
    emb_end_token = '</answer>'

    config = ConfigClass.from_pretrained(model_name)
    config.use_cache = False
    config.pad_token_id = tokenizer.pad_token_id
    tokenizer.save_pretrained(output_dir)

    ################### set model ###################

    base_model = ModelClass.from_pretrained(model_name,
                                            torch_dtype=torch.bfloat16,
                                            # attn_implementation="flash_attention_2",
                                            device_map={
                                                "": accelerator.process_index},
                                            config=config)

    ################### set generation ###################

    gen_config = GenerationConfig.from_pretrained(model_name)
    gen_config.max_new_tokens = max_new_tokens
    gen_config.num_return_sequences = group_size
    # Keep the training model cache disabled, but reuse KV states during
    # no-grad Hugging Face generation when vLLM is not used.
    gen_config.use_cache = True
    gen_config.top_k = gen_top_k
    gen_config.top_p = gen_top_p
    gen_config.temperature = gen_temperature

    ################################################################


    peft_config_dict = {
        "inference_mode": False,

        "target_modules": [
            'k_proj', 'v_proj', 'q_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj'
        ],
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }

    peft_config_dict.update(lora_kwargs)

    if use_lora:
        if init_checkpoint is not None:
            if not os.path.isdir(init_checkpoint):
                raise FileNotFoundError(
                    f"LoRA init checkpoint does not exist: {init_checkpoint}"
                )
            from peft import PeftModel

            base_model = PeftModel.from_pretrained(
                base_model,
                init_checkpoint,
                is_trainable=(not evaluation_only),
            )
            if accelerator.is_main_process:
                rich.print(
                    f"Loaded trainable LoRA checkpoint: {init_checkpoint}"
                )
        else:
            lora_cfg = {"r": 4, "lora_alpha": 128, }
            lora_cfg.update(peft_config_dict)
            peft_config = LoraConfig(**lora_cfg)
            if accelerator.is_main_process:
                rich.print(peft_config)
            base_model = get_peft_model(base_model, peft_config)
    else:
        if init_checkpoint is not None:
            raise ValueError(
                "init_checkpoint requires use_lora=True."
            )
        if accelerator.is_main_process:
            rich.print("No PEFT applied, training the base model")

    # base_model.enable_input_require_grads()
    ################### set trainer ###################
    # Derive the evaluation/save interval from the actual training world size.
    eval_steps = len(dset['train']) / (
        train_batch_size
        * gradient_accumulation_steps
        * accelerator.num_processes
    )
    eval_steps = max(1, int(eval_steps // eval_freq))

    training_args = OutcomeGRPOTrainingArguments(
        seed=seed,

        item_emb_batch_size=item_emb_batch_size,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=False,
        max_grad_norm=1.,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        bf16=True,

        save_strategy=("epoch" if fixed_epoch_combined_eval else "steps"),
        save_steps=eval_steps,
        save_only_model=False,
        # Keep every fixed-epoch checkpoint. Each one is written before that
        # epoch's expensive evaluation and its Trainer state is refreshed when
        # evaluation succeeds, so a failed evaluation remains resumable.
        save_total_limit=(None if fixed_epoch_combined_eval else 5),
        load_best_model_at_end=(not fixed_epoch_combined_eval),

        eval_strategy=("epoch" if fixed_epoch_combined_eval else "steps"),
        eval_steps=eval_steps,

        bf16_full_eval=True,
        per_device_eval_batch_size=eval_batch_size,
        metric_for_best_model=(
            None if fixed_epoch_combined_eval else 'eval_valid_ndcg@10'
        ),
        eval_on_start=(
            fixed_resume_needs_eval
            if fixed_epoch_combined_eval
            else eval_on_start
        ),
        batch_eval_metrics=True,

        logging_steps=1,
        output_dir=output_dir,
        optim="paged_adamw_8bit",
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
        report_to='none',
        run_name=run_name,
        gradient_checkpointing_kwargs={'use_reentrant': False},

        ddp_find_unused_parameters=False,
        remove_unused_columns=False,

        gather_negs_across_processes=gather_negs_across_processes,
        generation_config=gen_config,

        train_generation_batch_size=train_generation_batch_size,
        test_generation_batch_size=test_generation_batch_size,

        dataset_window_size=window_size,
        dataset_category=dataset_category,
        emb_token=emb_token,
        emb_end_token=emb_end_token,
        use_vllm=use_vllm,
        fixed_epoch_combined_eval=fixed_epoch_combined_eval,
        **trainer_extra_kwargs,

    )
    metric_updater = MetricUpdater(ks=[5, 10, 20])

    trainer = OutcomeGRPOTrainer(
        model=base_model,
        compute_metrics=get_compute_metrics(metric_updater, ),
        data_collator=DataCollator(tokenizer=tokenizer,
                                   return_tensors="pt"),

        full_dataset=dset,

        callbacks=([] if fixed_epoch_combined_eval else [
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience
            )
        ]) + [PeriodicCheckpointCallback()],
        processing_class=tokenizer,
        args=training_args,
    )

    if fixed_epoch_combined_eval and accelerator.is_main_process:
        sizes = trainer.combined_eval_split_sizes
        if evaluation_only:
            rich.print(
                "Checkpoint-only combined evaluation enabled: "
                f"valid={sizes['original_valid_rows']}, "
                f"test={sizes['original_test_rows']}, "
                f"combined={sizes['combined_test_rows']}. "
                "No optimizer step will run."
            )
        else:
            rich.print(
                "Fixed-epoch combined evaluation enabled: "
                f"valid={sizes['original_valid_rows']}, "
                f"test={sizes['original_test_rows']}, "
                f"combined={sizes['combined_test_rows']}. "
                f"Evaluating eval_test_* after each of {int(num_train_epochs)} "
                "epochs, retaining each epoch-boundary checkpoint, and "
                f"saving an additional resumable checkpoint every "
                f"{_PERIODIC_CHECKPOINT_STEPS} steps."
            )
        if resume_from_checkpoint:
            rich.print(
                f"Resuming from {resume_from_checkpoint}; "
                f"evaluate_on_resume={fixed_resume_needs_eval}."
            )

    if evaluation_only:
        evaluation_metrics = trainer.evaluate()
        evaluation_metrics.update(trainer.combined_eval_split_sizes)
        trainer.log_metrics("candidate_eval", evaluation_metrics)
        trainer.save_metrics("candidate_eval", evaluation_metrics)

        candidate_manifest_writer = getattr(
            trainer,
            "_write_candidate_protocol_manifest",
            None,
        )
        if callable(candidate_manifest_writer):
            candidate_manifest_writer(
                metrics_file="candidate_eval_results.json",
                execution_mode="checkpoint_only",
                evaluation_schedule="once",
                training_protocol="none",
                evaluated_checkpoint=init_checkpoint,
            )

        if accelerator.is_main_process:
            protocol_manifest = {
                "protocol": "checkpoint_only_combined_eval",
                "evaluated_checkpoint": os.path.abspath(init_checkpoint),
                "metric_prefix": "eval_test",
                "evaluation_rows": "valid_plus_test",
                "generation_max_new_tokens": int(max_new_tokens),
                "metrics_file": "candidate_eval_results.json",
                **trainer.combined_eval_split_sizes,
            }
            with open(
                    os.path.join(output_dir, "checkpoint_eval_protocol.json"),
                    "w",
                    encoding="utf-8",
            ) as handle:
                json.dump(
                    protocol_manifest,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
        return evaluation_metrics

    train_result = trainer.train(
        resume_from_checkpoint=resume_from_checkpoint
    )
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    if fixed_epoch_combined_eval:
        final_test_metrics = last_prefixed_metrics(
            trainer.state.log_history,
            prefix="eval_test_",
        )
        if final_test_metrics is None:
            # Defensive fallback for an unusual Trainer exit that skipped the
            # scheduled final-epoch evaluation.
            final_test_metrics = trainer.evaluate()
        final_test_metrics.update(trainer.combined_eval_split_sizes)
        final_test_metrics["final_global_step"] = int(
            trainer.state.global_step
        )
        trainer.log_metrics("final_test", final_test_metrics)
        trainer.save_metrics("final_test", final_test_metrics)

        if accelerator.is_main_process:
            protocol_manifest = {
                "protocol": "fixed_epoch_combined_eval",
                "evaluation_schedule": "epoch",
                "metric_prefix": "eval_test",
                "num_train_epochs": int(num_train_epochs),
                **trainer.combined_eval_split_sizes,
                "final_global_step": int(trainer.state.global_step),
                "final_metrics_file": "final_test_results.json",
                "trainer_state_file": "trainer_state.json",
                "checkpoint_policy": "epoch_boundaries_plus_periodic_steps",
                "periodic_checkpoint_steps": (
                    _PERIODIC_CHECKPOINT_STEPS
                ),
                "resumed_from_checkpoint": (
                    str(resume_from_checkpoint)
                    if resume_from_checkpoint
                    else None
                ),
            }
            os.makedirs(output_dir, exist_ok=True)
            with open(
                    os.path.join(output_dir, "fixed_epoch_protocol.json"),
                    "w",
                    encoding="utf-8",
            ) as handle:
                json.dump(
                    protocol_manifest,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )

    if cleanup_previous_checkpoints:
        os.system(f"rm -rf {output_dir}/checkpoint-*")
        print(f"Removed previous checkpoints in {output_dir}")

    final_output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.save_model(final_output_dir)
    trainer.save_state()
    if trainer.is_world_process_zero():
        trainer.state.save_to_json(
            os.path.join(final_output_dir, "trainer_state.json")
        )


if __name__ == "__main__":
    import fire

    fire.Fire(train)
