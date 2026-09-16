"""U-GAP runtime components extracted from the research implementation."""

import re

def parse_mcq_prediction(response):
    """Parse an option label without consulting the gold answer."""
    text = str(response or "").strip()
    normalized = text.replace('"', "").replace("'", "")
    match = re.search(r"\bAnswer\s*:\s*([A-E])\b", normalized, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    if re.fullmatch(r"\s*([A-E])\s*", normalized, re.IGNORECASE):
        return normalized.strip().upper()
    return "None"
