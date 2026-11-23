import json
import re
import ast
from pathlib import Path


def chunk_text(text, size=2000):
    chunks = []

    for i in range(0, len(text), size):
        chunks.append(text[i:i + size])

    return chunks


def chunk_python(text, size=2000):
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return chunk_text(text, size)

    lines = text.splitlines(keepends=True)
    starts = [0, len(lines)]

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first_line = node.decorator_list[0].lineno if node.decorator_list else node.lineno
            starts.append(first_line - 1)

    starts = sorted(set(starts))
    chunks = []
    current = ""

    for start, end in zip(starts, starts[1:]):
        section = "".join(lines[start:end])

        if len(section) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(chunk_text(section, size))

        elif len(current) + len(section) <= size:
            current += section

        else:
            chunks.append(current)
            current = section

    if current:
        chunks.append(current)

    return chunks


def chunk_markdown(text, size=2000):
    sections = []
    current = ""

    for line in text.splitlines(keepends=True):
        if re.match(r"^ {0,3}#{1,6}(?:[ \t]|$)", line) and current:
            sections.append(current)
            current = ""

        current += line

    if current:
        sections.append(current)

    chunks = []
    current = ""

    for section in sections:
        if len(section) > size:
            if current:
                chunks.append(current)
                current = ""

            chunks.extend(chunk_text(section, size))

        elif len(current) + len(section) <= size:
            current += section

        else:
            chunks.append(current)
            current = section

    if current:
        chunks.append(current)

    return chunks

root = Path("data/raw/vllm-0.10.1")

if not root.is_dir():
    raise SystemExit("vLLM folder not found")

indexed_chunks = []

for path in sorted(root.rglob("*")):
    if not path.is_file() or path.suffix not in (".py", ".md"):
        continue

    content = path.read_text(
        encoding="utf-8",
        errors="replace"
    )

    if path.suffix == ".py":
        chunks = chunk_python(content)
    else:
        chunks = chunk_markdown(content)

    start = 0

    for chunk in chunks:
        end = start + len(chunk)

        if content[start:end] != chunk:
            raise ValueError(f"Invalid chunk position: {path}")

        indexed_chunks.append({
            "file_path": path.as_posix(),
            "first_character_index": start,
            "last_character_index": end,
            "text": chunk
        })

        start = end

    if start != len(content):
        raise ValueError(f"Missing characters: {path}")

output = Path("data/processed/chunks.json")
output.parent.mkdir(parents=True, exist_ok=True)

with output.open("w", encoding="utf-8") as file:
    json.dump(indexed_chunks, file, ensure_ascii=False)

print("Total chunks:", len(indexed_chunks))
print("Saved to:", output)
