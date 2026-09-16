"""U-GAP runtime components extracted from the research implementation."""

import hashlib

import json

import re

from src.promptTemplate import answer_generation_direct_prompt_template

def compact(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()

def sha256_text(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()

def neutralize_information_need(text):
    """Remove option-letter framing while retaining the medical concepts."""
    value = compact(text)
    value = re.sub(r"\b(?:answer\s+)?(?:option|choice)\s*\(?[A-E]\)?\b", "", value,
                   flags=re.IGNORECASE)
    value = re.sub(r"\b(?:select|choose)\s+(?:the\s+)?(?:option|choice)\b", "determine the medical principle",
                   value, flags=re.IGNORECASE)
    value = re.sub(r"\bcorrect\s+answer\b", "relevant medical conclusion", value,
                   flags=re.IGNORECASE)
    value = re.sub(r"\s+([,.;:])", r"\1", value)
    value = compact(value).strip(" -:;")
    return value

def build_generator_prompt(question, generation_information_need, max_tokens=256):
    # Deliberately accepts no options argument.
    return f"""You are a medical knowledge generator.

Given a medical question and a specific missing information need,
generate a concise medical background passage that addresses ONLY
the specified missing medical knowledge.

Requirements:
- Focus only on the provided information need.
- State established medical facts, associations, mechanisms,
  diagnostic features, or treatment principles when relevant.
- Do not answer the multiple-choice question.
- Do not mention answer choices or option letters.
- Do not speculate about the correct answer.
- Do not invent unsupported details.
- Keep the passage concise and self-contained.
- Maximum length: {int(max_tokens)} tokens.

Question:
{compact(question)}

Missing Information Need:
{compact(generation_information_need)}

Background Knowledge:
"""

def clean_generated_background(text):
    value = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.DOTALL | re.IGNORECASE)
    value = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", value.strip())
    value = re.sub(r"^Background Knowledge\s*:\s*", "", value, flags=re.IGNORECASE)
    return value.strip()

def _insert_before_question(prompt, block):
    marker = "\nQuestion:\n"
    if prompt.count(marker) != 1:
        raise ValueError("Reader prompt template no longer has exactly one Question section")
    return prompt.replace(marker, "\n" + block.strip() + "\n\nQuestion:\n", 1)

def build_generation_only_reader_prompt(answer, query, generated_background):
    instruction = answer._build_answer_instruction(query)
    prompt = (
        answer_generation_direct_prompt_template
        .replace("{q}", query)
        .replace("{answer_instruction}", instruction)
    )
    generated = (
        "Generated Background Knowledge "
        "(auxiliary model-generated context; it may be incomplete):\n"
        + str(generated_background).strip()
    )
    return _insert_before_question(prompt, generated)

def record_sha256(record):
    serialized = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(serialized)
