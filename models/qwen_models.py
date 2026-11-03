from typing import Optional, Union, List

import torch
from torch.nn import Linear, Embedding
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from transformers.cache_utils import HybridCache

from models.abstract_models import AbsModelConfig, AbsModel


class Qwen2RecommendationConfig(Qwen2Config, AbsModelConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Qwen2RecommendationCasualLM(Qwen2ForCausalLM):
    config_class = Qwen2RecommendationConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = Linear(config.hidden_size,
                              config.vocab_size, bias=False)
        self.post_init()

    def set_token_embedding(self, idx_to_paste: int, idx_to_copy: List[int]):
        self.model.embed_tokens.weight.data[idx_to_paste] = self.model.embed_tokens.weight.data[
            idx_to_copy].mean(dim=0)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def _sequence_hidden_with_last_layer_attention(
            self,
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            use_cache,
            cache_position,
    ):
        """Return sequence hidden states and only the final decoder attention.

        Hugging Face exposes one global ``output_attentions`` flag, which
        otherwise materialises an ``S x S`` map for every decoder layer.  OPSD
        needs only the final Qwen layer, so a scoped hook changes that layer's
        local flag while the backbone-wide flag remains disabled.  The OPSD
        caller always uses ``use_cache=False``; retaining that restriction also
        keeps the outer Qwen layer-output indexing unchanged.
        """

        if use_cache:
            raise ValueError(
                "Last-layer attention capture requires use_cache=False."
            )
        layers = getattr(self.model, "layers", None)
        if layers is None or len(layers) == 0:
            raise RuntimeError(
                "Qwen backbone does not expose decoder layers for attention "
                "capture."
            )

        captured = {}

        def _force_final_layer_attention(module, args, kwargs):
            kwargs = dict(kwargs)
            kwargs["output_attentions"] = True
            return args, kwargs

        def _capture_final_layer_attention(module, args, kwargs, output):
            if not isinstance(output, (tuple, list)) or len(output) < 2:
                raise RuntimeError(
                    "Final Qwen decoder layer did not return attention weights."
                )
            captured["attention"] = output[1]

        final_layer = layers[-1]
        pre_hook = final_layer.register_forward_pre_hook(
            _force_final_layer_attention,
            with_kwargs=True,
        )
        post_hook = final_layer.register_forward_hook(
            _capture_final_layer_attention,
            with_kwargs=True,
        )
        try:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                cache_position=cache_position,
            )
        finally:
            pre_hook.remove()
            post_hook.remove()

        last_layer_attention = captured.get("attention")
        if last_layer_attention is None:
            raise RuntimeError("Failed to capture final-layer Qwen attention.")
        if attention_mask is None:
            readout_positions = torch.full(
                (last_layer_attention.shape[0],),
                last_layer_attention.shape[-2] - 1,
                device=last_layer_attention.device,
                dtype=torch.long,
            )
        else:
            if attention_mask.ndim != 2:
                raise ValueError(
                    "Last-layer attention capture expects a rank-2 padding mask."
                )
            valid_mask = attention_mask.to(
                device=last_layer_attention.device,
                dtype=torch.bool,
            )
            positions = torch.arange(
                valid_mask.shape[1],
                device=valid_mask.device,
            ).unsqueeze(0).expand_as(valid_mask)
            readout_positions = positions.masked_fill(
                ~valid_mask,
                -1,
            ).max(dim=1).values
            if (readout_positions < 0).any():
                raise ValueError("Cannot capture attention for an empty sequence.")
        batch_indices = torch.arange(
            last_layer_attention.shape[0],
            device=last_layer_attention.device,
        )
        readout_attention = last_layer_attention[
            batch_indices,
            :,
            readout_positions,
            :,
        ]
        return outputs.last_hidden_state, readout_attention

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[HybridCache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            return_causal_output: bool = True,
            return_with_last_hidden_states: bool = False,
            return_sequence_hidden_states: bool = False,
            return_last_layer_attention: bool = False,
            **loss_kwargs,

    ):

        assert (
            return_causal_output
            or return_with_last_hidden_states
            or return_sequence_hidden_states
        ), (
            "At least one model output mode must be enabled."
        )
        if return_sequence_hidden_states:
            if return_causal_output or return_with_last_hidden_states:
                raise ValueError(
                    "return_sequence_hidden_states is a standalone output mode."
                )
            if return_last_layer_attention:
                if output_attentions not in (None, False):
                    raise ValueError(
                        "Use return_last_layer_attention without enabling "
                        "global output_attentions."
                    )
                return self._sequence_hidden_with_last_layer_attention(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=False,
                return_dict=True,
                cache_position=cache_position,
            )
            return outputs.last_hidden_state

        if return_last_layer_attention:
            raise ValueError(
                "return_last_layer_attention requires "
                "return_sequence_hidden_states=True."
            )

        if not return_causal_output:
            result = AbsModel.forward(self,
                                            attention_mask=attention_mask,
                                            input_ids=input_ids,
                                            position_ids=position_ids, )
            return result

        if return_with_last_hidden_states:
            output_hidden_states = True
            return_dict = True

        causallm_output = Qwen2ForCausalLM.forward(self,
                                                   input_ids=input_ids,
                                                   attention_mask=attention_mask,
                                                   position_ids=position_ids,
                                                   past_key_values=past_key_values,
                                                   inputs_embeds=inputs_embeds,
                                                   labels=labels,
                                                   use_cache=use_cache,
                                                   output_attentions=output_attentions,
                                                   output_hidden_states=output_hidden_states,
                                                   return_dict=return_dict,
                                                   cache_position=cache_position,
                                                   logits_to_keep=logits_to_keep,
                                                   **loss_kwargs,
                                                   )

        if not return_with_last_hidden_states:
            return causallm_output

        # tuple(torch.FloatTensor)[-1] ->  (B, S, hidden_size)
        last_block_hidden_states = causallm_output.hidden_states[-1]

        # (B, hidden_size)
        last_token_hidden_states = AbsModel.forward(self,
                                                          attention_mask=attention_mask,
                                                          input_ids=input_ids,
                                                          hidden_states=last_block_hidden_states,
                                                          )

        return causallm_output, last_token_hidden_states
