from __future__ import annotations

import os

# Set this to False once when you need to download models from Hugging Face.
# After models are cached locally, keep it True for fully local runs.
USE_LOCAL_MODEL_CACHE_ONLY = True

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
VLM_MODEL_PATH = "mlx-community/Qwen2-VL-7B-Instruct-4bit"


def apply_model_cache_policy() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    if USE_LOCAL_MODEL_CACHE_ONLY:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
