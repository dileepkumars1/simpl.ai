# bc-12-put-a-face-on-it.py — putting a face on it
# Run with: python bc-12-put-a-face-on-it.py
# First install gradio: pip install gradio

import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

def respond(message, history):
    # `history` is a list of {"role": ..., "content": ...} dicts Gradio maintains for us —
    # this is exactly bc-05's chat loop, just fed by a UI instead of a Python list you append to.
    messages = history + [{"role": "user", "content": message}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

demo = gr.ChatInterface(
    fn=respond,
    type="messages",
    title="Your Local Chatbot",
    description="Running entirely on this machine — no API key, no network call.",
)

if __name__ == "__main__":
    demo.launch()
