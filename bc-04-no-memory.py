# bc-04-no-memory.py — why it forgot what you just said
# Run with: python bc-04-no-memory.py

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

def ask(message):
    # Every call starts from scratch — nothing about a previous call is passed in here.
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": message}], tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    outputs = model.generate(**inputs, max_new_tokens=30)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

print("Turn 1 user: My name is Alex.")
print("Turn 1 reply:", ask("My name is Alex."))

print()
print("Turn 2 user (fresh call, no history passed in): What's my name?")
print("Turn 2 reply:", ask("What's my name?"))
