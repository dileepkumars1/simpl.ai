# bc-01-ten-lines.py — the smallest possible chatbot (and why it's bad)
# Run with: python bc-01-ten-lines.py

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

prompt = "Hi, how are you?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=30)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# Notice: no "chat" formatting anywhere above — just raw text in, raw text out.
# That's exactly why the output below doesn't read like a reply. See the page for why.
