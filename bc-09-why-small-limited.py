# bc-09-why-small-limited.py — why "small" means limited, on screen
# Run with: python bc-09-why-small-limited.py

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

def ask(message, max_new_tokens=50):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": message}], tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

print("ARITHMETIC:")
print(ask("What is 47 plus 89?"))
print()

print("A CONFIDENTLY WRONG FACT:")
print(ask("Who was the first person to climb K2 alone in winter?"))
print()

print("SELF-CONTRADICTION (two separate, unrelated calls):")
print("Q1:", ask("Do you have feelings?"))
print("Q2:", ask("Are you just a program with no feelings at all?"))
