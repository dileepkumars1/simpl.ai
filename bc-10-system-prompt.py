# bc-10-system-prompt.py — the system prompt, in code
# Run with: python bc-10-system-prompt.py

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

def ask(messages, max_new_tokens=50):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    reply = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return prompt, reply

question = "Should I invest my savings in cryptocurrency?"

print("---- no system message written by you ----")
prompt1, reply1 = ask([{"role": "user", "content": question}])
print("assembled prompt:", repr(prompt1))
print("reply:", reply1)
print()

print("---- with an explicit system message ----")
system_msg = ("You are a cautious, formal financial-literacy assistant. You never give direct "
              "investment advice; you explain tradeoffs and always suggest consulting a licensed advisor.")
prompt2, reply2 = ask([{"role": "system", "content": system_msg}, {"role": "user", "content": question}])
print("assembled prompt:", repr(prompt2))
print("reply:", reply2)
