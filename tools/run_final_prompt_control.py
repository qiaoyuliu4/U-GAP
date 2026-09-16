"""U-GAP runtime components extracted from the research implementation."""

from src.promptTemplate import answer_generation_direct_prompt_template

def call_reader(answer, query, formatted, arm):
    # Refuse to drop real non-text sources: this ablation is text-only MedText.
    if any(formatted[k] for k in ("selected_kg", "alignment_notes", "conflict_notes")):
        raise ValueError("Non-text sources/notes present; cannot isolate a text-only prompt control")
    if arm == "rag":
        return answer.call_selected_evidence(query, **formatted)
    if arm != "direct":
        raise ValueError("Unknown prompt arm: " + arm)
    marker = "Question:\n{q}"
    if answer_generation_direct_prompt_template.count(marker) != 1:
        raise ValueError("Direct template changed; review evidence insertion point")
    template = answer_generation_direct_prompt_template.replace(
        marker, "Retrieved medical evidence:\n{final_text}\n\n" + marker)
    prompt = template.replace("{answer_instruction}", answer._build_answer_instruction(query))
    prompt = prompt.replace("{q}", query).replace("{final_text}", formatted["selected_text"])
    return answer._decode(prompt, query, new_tokens_num=answer.args.answer_tokens)
