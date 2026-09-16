"""U-GAP runtime components extracted from the research implementation."""

from src.parametric_generation_diagnostic import build_generation_only_reader_prompt, clean_generated_background, neutralize_information_need, record_sha256

TARGET_ROUTE = "retrieve"

TARGET_STOP_REASON = "one_followup_limit_reached"

def is_target_record(row):
    return (
        row.get("route") == TARGET_ROUTE
        and row.get("stop_reason") == TARGET_STOP_REASON
    )

def _usable_missing_relations(value):
    return [
        gap
        for gap in (value or [])
        if isinstance(gap, dict) and str(gap.get("relation", "")).strip()
    ]

def extract_final_generation_information_need(row):
    """Read only a unified final need or final ESA missing relations.

    Planner information_needs and pre-follow-up gaps are deliberately excluded.
    """
    unified = str(row.get("generation_information_need", "") or "").strip()
    if unified:
        neutral = neutralize_information_need(unified)
        if neutral:
            return {
                "generation_information_need": neutral,
                "generation_information_need_source": "generation_information_need",
                "final_missing_relations": [],
            }

    state = row.get("evidence_state") or {}
    gaps = _usable_missing_relations(state.get("missing_relations"))
    if not gaps:
        return None

    rendered = []
    seen = set()
    for gap in gaps:
        relation = neutralize_information_need(gap.get("relation"))
        why_needed = neutralize_information_need(gap.get("why_needed"))
        text = relation
        if why_needed and why_needed.lower() not in relation.lower():
            text += ". Clinical relevance: " + why_needed
        text = text.strip().rstrip(".")
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            rendered.append(text + ".")

    if not rendered:
        return None
    return {
        "generation_information_need": " ".join(rendered),
        "generation_information_need_source": "evidence_state.missing_relations",
        "final_missing_relations": gaps,
    }
