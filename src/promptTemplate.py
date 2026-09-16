"""U-GAP runtime components extracted from the research implementation."""

answer_generation_snippets_only_prompt_template = """
The following is a multiple-choice medical question with retrieved medical snippets from external medical corpora.

Use the retrieved snippets as supporting medical context. The snippets may be incomplete or partially irrelevant, so prioritize the question, the answer options, and established medical knowledge when deciding.

{answer_instruction}

Retrieved medical snippets:
{snippets}

Question:
{q}
"""

answer_generation_direct_prompt_template = """
The following is a multiple-choice medical question.

Answer using established medical knowledge.

{answer_instruction}

Question:
{q}
"""

answer_generation_selected_evidence_prompt_template = """
The following is a medical question with evidence selected from a dual-source retrieval pipeline.

Evidence sources:
1. Selected text evidence from medical corpora.
2. Selected UMLS/SemMedDB literature KG evidence, when available.
3. Alignment notes showing whether text and KG evidence support or complement each other.
4. Conflict notes, when weak conflicts are detected.

Use selected text evidence as the primary evidence. Use KG evidence only when it is aligned with the question or complements the text evidence.
Ignore KG evidence that is too generic, weakly aligned, or contradicted by stronger text evidence.
If the selected evidence is sparse or inconclusive, use established medical knowledge while respecting the answer options.

Available answer options:
{option_lines}

Task rules:
{task_rules}

Selected text evidence:
{selected_text}

Selected KG evidence:
{selected_kg}

Text-KG alignment notes:
{alignment_notes}

Conflict notes:
{conflict_notes}

Question:
{q}
"""
