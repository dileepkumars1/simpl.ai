# bc-00-setup.py — the smoke test from "Your Local AI Lab"
# Run with: python bc-00-setup.py
# First run downloads ~270MB to your Hugging Face cache — that's expected, see the page.

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

inputs = tokenizer("The capital of France is", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0]))
