
import json
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


chunks_path = Path("data/processed/chunks.json")

with chunks_path.open("r", encoding="utf-8") as file:
    chunks = json.load(file)

if not chunks:
    raise SystemExit("No chunks found")

tokenized_chunks = []

for chunk in chunks:
    tokens = re.findall(r"[a-z0-9]+", chunk["text"].lower())
    tokenized_chunks.append(tokens)

bm25 = BM25Okapi(tokenized_chunks)

output = Path("data/processed/bm25.pkl")

with output.open("wb") as file:
    pickle.dump(bm25, file)

print("Total indexed chunks:", len(chunks))
print("BM25 index saved:", output)
