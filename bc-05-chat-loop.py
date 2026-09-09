# bc-05-chat-loop.py — the chat loop, with real memory
# Run with: python bc-05-chat-loop.py

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

history = []  # every turn, user and assistant, accumulates here

def chat(user_message):
    history.append({"role": "user", "content": user_message})
    prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    print("---- full assembled prompt sent to the model ----")
    print(prompt)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    outputs = model.generate(**inputs, max_new_tokens=30)
    reply = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    history.append({"role": "assistant", "content": reply})
    return reply

print("Reply 1:", chat("My name is Alex."))
print()
print("Reply 2:", chat("What's my name?"))
