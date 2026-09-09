# bc-03-logits-to-word.py — from a wall of numbers to one word
# Run with: python bc-03-logits-to-word.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

prompt = "My favorite season is"
inputs = tokenizer(prompt, return_tensors="pt")

# One forward pass, no generation yet — just the raw scores for "what word comes next"
with torch.no_grad():
    logits = model(**inputs).logits[0, -1]   # logits for the very next token, shape (vocab_size,)

print("logits shape:", tuple(logits.shape))
top5 = torch.topk(logits, 5)
print("top-5 raw logits:")
for val, idx in zip(top5.values.tolist(), top5.indices.tolist()):
    print(f"  {val:7.3f}  {repr(tokenizer.decode([idx]))}")

probs = torch.softmax(logits, dim=-1)
top5p = torch.topk(probs, 5)
print("top-5 probabilities (after softmax):")
for val, idx in zip(top5p.values.tolist(), top5p.indices.tolist()):
    print(f"  {val:.4f}  {repr(tokenizer.decode([idx]))}")

# Now vary temperature and actually sample from this same distribution
for temp in [0.2, 1.0, 1.5]:
    out_ids = model.generate(**inputs, max_new_tokens=15, do_sample=True, temperature=temp, top_k=0, top_p=1.0)
    reply = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"temperature={temp}: {reply!r}")
