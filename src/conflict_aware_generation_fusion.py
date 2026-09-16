"""U-GAP runtime components extracted from the research implementation."""

from src.promptTemplate import answer_generation_direct_prompt_template

def build_direct_conflict_aware_fusion_prompt(
    answer,
    query,
    selected_text,
    information_need,
    generated_background,
):
    """Use the tested Direct instructions plus two explicitly separated sources."""
    marker = "Question:\n{q}"
    if answer_generation_direct_prompt_template.count(marker) != 1:
        raise ValueError("Direct template changed; review fusion insertion point")
    evidence_block = f"""Retrieved medical evidence:
{str(selected_text).strip() or "No retrieved medical evidence was provided."}

Unresolved information need:
{str(information_need).strip()}

Unverified generated background knowledge:
{str(generated_background).strip()}

Source and decision rules:
- Retrieved evidence comes from medical corpora but may be incomplete.
- Generated background is model-generated, unverified, and may contain errors.
- Use generated knowledge only when it directly addresses the unresolved need and is medically consistent with the question and reliable retrieved evidence.
- If generated knowledge conflicts with clearly supported retrieved evidence, prefer the retrieved evidence.
- Do not interpret absence of supporting evidence as evidence that a statement is false.
- Re-evaluate the question from the evidence and established medical knowledge. Previous pipeline predictions are not provided and must not be inferred."""
    template = answer_generation_direct_prompt_template.replace(
        marker,
        evidence_block + "\n\n" + marker,
    )
    return (
        template
        .replace("{answer_instruction}", answer._build_answer_instruction(query))
        .replace("{q}", query)
    )
