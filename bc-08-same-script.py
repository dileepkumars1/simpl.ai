# bc-08-same-script.py — same script, one line changed
# Run with: python bc-08-same-script.py

from transformers import AutoTokenizer, AutoModelForCausalLM

# The only thing that changes between runs is this one string.
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
# MODEL_ID = "distilbert/distilgpt2"
# MODEL_ID = "openai-community/gpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

prompt = "Hi, how are you?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=30)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
