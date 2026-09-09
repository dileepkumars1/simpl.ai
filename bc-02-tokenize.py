# bc-02-tokenize.py — what tokenize() really returns
# Run with: python bc-02-tokenize.py

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

sentence = "Tokenization splits unfamiliar words into pieces."
ids = tokenizer(sentence)["input_ids"]

print("sentence:", sentence)
print("token ids:", ids)
print("count:", len(ids))
for i in ids:
    print(f"  {i:6d} -> {repr(tokenizer.decode([i]))}")
