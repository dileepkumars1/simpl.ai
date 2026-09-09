# bc-07-model-comparison.py — a model isn't one thing
# Run with: python bc-07-model-comparison.py
# Only fetches each model's config.json (a few KB) — no full weights downloaded here.

from transformers import AutoConfig

MODELS = [
    ("distilbert/distilgpt2", "base", 81_912_576),
    ("openai-community/gpt2", "base", 124_439_808),
    ("HuggingFaceTB/SmolLM2-135M-Instruct", "instruct", 134_515_008),
    ("HuggingFaceTB/SmolLM2-360M-Instruct", "instruct", None),
    ("Qwen/Qwen2.5-0.5B-Instruct", "instruct", None),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "instruct", None),
    ("microsoft/Phi-3-mini-4k-instruct", "instruct", None),
]

print(f"{'model':42s} {'kind':10s} {'ctx (tokens)':13s} {'layers':7s} {'hidden':7s}")
for model_id, kind, known_params in MODELS:
    cfg = AutoConfig.from_pretrained(model_id)
    print(f"{model_id:42s} {kind:10s} {cfg.max_position_embeddings:<13d} {cfg.num_hidden_layers:<7d} {cfg.hidden_size:<7d}")
