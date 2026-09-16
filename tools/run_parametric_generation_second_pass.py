"""U-GAP runtime components extracted from the research implementation."""

import hashlib

import json

import os

from pathlib import Path

from src.parametric_generation_second_pass import record_sha256

def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows

def index_rows(rows, name):
    indexed = {}
    for row in rows:
        if "index" not in row:
            raise ValueError(f"Missing index in {name}")
        index = int(row["index"])
        if index in indexed:
            raise ValueError(f"Duplicate index {index} in {name}")
        indexed[index] = row
    return indexed

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def write_json_atomic(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)

def append_jsonl(path, row):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

class LoggedLLM:
    """Capture the exact prompt while delegating to one shared BaseLLM instance."""

    def __init__(self, delegate):
        self.delegate = delegate
        self.llm_name = delegate.llm_name
        self.last_prompt = None

    def generate(self, query, new_tokens_num):
        self.last_prompt = query
        return self.delegate.generate(query, new_tokens_num)
