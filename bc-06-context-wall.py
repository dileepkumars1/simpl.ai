# bc-06-context-wall.py — hitting the wall on purpose
# Run with: python bc-06-context-wall.py

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

print("this model's context limit:", model.config.max_position_embeddings, "tokens")

# Build a conversation deliberately longer than that limit.
huge_message = "word " * 9000
history = [{"role": "user", "content": huge_message}]
prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

print("this conversation is", inputs["input_ids"].shape[1], "tokens long")
outputs = model.generate(**inputs, max_new_tokens=5)
print("generate() returned a result anyway — shape:", tuple(outputs.shape))
print("but read the warnings above: it explicitly says results past the limit aren't reliable.")
