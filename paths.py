"""Public model locations with environment overrides."""

import os


model_names = {
    "Gemma-2-2b-it": os.environ.get(
        "GEMMA_MODEL_NAME_OR_PATH",
        "google/gemma-2-2b-it",
    ),
    "Qwen2.5-3B-Instruct": os.environ.get(
        "MODEL_NAME_OR_PATH",
        "Qwen/Qwen2.5-3B-Instruct",
    ),
    "Qwen2.5-0.5B-Instruct": os.environ.get(
        "QWEN_SMALL_MODEL_NAME_OR_PATH",
        "Qwen/Qwen2.5-0.5B-Instruct",
    ),
}
