# bc-14-end-to-end.py — the whole thing, end to end
# Everything from Modules 1-4, stitched into one file.
# Run with: python bc-14-end-to-end.py
# For the Gradio UI at the bottom: pip install gradio, then uncomment demo.launch()

import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- Module 1: load the model (bc-01) ----
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

# ---- Module 4: a system prompt, written on purpose (bc-10) ----
SYSTEM_PROMPT = "You are a helpful, concise local assistant running entirely on the user's own machine."

# ---- Module 4: a guardrail you can actually read (bc-11) ----
BLOCKED_PATTERNS = [
    r"\bhow to (make|build) a bomb\b",
    r"\bmy (password|social security number) is\b",
]

def is_blocked(text):
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in BLOCKED_PATTERNS)

# ---- Module 2: real memory, one growing list (bc-05) ----
history = [{"role": "system", "content": SYSTEM_PROMPT}]

def respond(message):
    if is_blocked(message):
        return "[BLOCKED before reaching the model]"
    history.append({"role": "user", "content": message})
    prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

    # ---- Module 2: hitting the wall on purpose (bc-06) — warn, don't silently proceed ----
    if inputs["input_ids"].shape[1] > model.config.max_position_embeddings:
        return "[conversation too long for this model's context window]"

    outputs = model.generate(**inputs, max_new_tokens=80)
    reply = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    history.append({"role": "assistant", "content": reply})
    return reply


if __name__ == "__main__":
    # Plain terminal loop — this is the whole series, minus the UI.
    print("Local chatbot ready. Type a message (Ctrl+C to quit).")
    print(respond("Hi! What can you help me with?"))

    # ---- Module 4: put a face on it (bc-12) ----
    # import gradio as gr
    # demo = gr.ChatInterface(fn=lambda message, hist: respond(message), type="messages")
    # demo.launch()
