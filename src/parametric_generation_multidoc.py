"""U-GAP runtime components extracted from the research implementation."""

from copy import deepcopy

from src.parametric_generation_diagnostic import build_generator_prompt, neutralize_information_need

from tools.run_parametric_generation_second_pass import LoggedLLM

DOCUMENTS_PER_QUESTION = 5

def document_specs(row, seed, document_count=5):
    """Keep final gap order; pad with independently sampled question backgrounds."""
    specs, seen = [], set()
    for gap in row.get("final_missing_relations", []):
        relation = neutralize_information_need(gap.get("relation", ""))
        if not relation or relation.casefold() in seen:
            continue
        seen.add(relation.casefold())
        specs.append({"gap_id": gap.get("id"), "kind": "final_gap",
                      "generation_information_need": relation})
    if not specs and row.get("generation_information_need"):
        specs.append({"gap_id": None, "kind": "final_gap",
                      "generation_information_need": row["generation_information_need"]})
    if not specs:
        raise ValueError(f"missing_gap: index={row['index']}")
    if document_count < 1:
        raise ValueError("document_count must be positive")
    if len(specs) > document_count:
        raise ValueError(f"More final gaps than document_count at index={row['index']}; increase the budget")
    while len(specs) < document_count:
        specs.append({"gap_id": None, "kind": "question_background",
                      "generation_information_need": None})
    for number, spec in enumerate(specs, 1):
        spec.update(document_number=number, document_id=f"G{number}",
                    seed=seed + int(row["index"]) * document_count + number - 1)
    return specs

def generation_prompt(question, spec):
    if spec["kind"] == "final_gap":
        return build_generator_prompt(question, spec["generation_information_need"], 256)
    return f"""You are a medical knowledge generator.

Given a medical question, generate a concise, self-contained background passage
about the established medical knowledge relevant to understanding the question.
Include relevant associations, mechanisms, diagnostic features, or principles.
Do not answer the question, mention option letters, or guess the correct answer.
Do not invent unsupported details. Maximum length: 256 tokens.

Question:
{question}

Background Knowledge:
"""

def render_documents(documents):
    return "\n\n".join(
        f"Generated Document [{doc['document_id']}]:\n{doc['generated_background']}"
        for doc in documents
    )

def chat_input_audit(llm, prompt, output_tokens, context_limit):
    ids = llm.llm_tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=True,
    )
    model_limit = int(llm.llm_model.config.max_position_embeddings)
    effective_limit = min(context_limit, model_limit)
    if len(ids) + output_tokens > effective_limit:
        raise ValueError(
            f"Context overflow: input={len(ids)}, output_budget={output_tokens}, "
            f"limit={effective_limit}. No documents were truncated."
        )
    return {"input_tokens": len(ids), "output_token_budget": output_tokens,
            "context_limit": effective_limit, "truncated": False}

class CheckedReaderLLM(LoggedLLM):
    def __init__(self, delegate, context_limit):
        super().__init__(delegate)
        self.context_limit = context_limit
        self.input_audit = None

    def generate(self, query, new_tokens_num):
        self.input_audit = chat_input_audit(
            self.delegate, query, new_tokens_num, self.context_limit,
        )
        return super().generate(query, new_tokens_num)

def sample_document(llm, prompt, spec, args):
    audit = chat_input_audit(llm, prompt, 256, args.context_limit)
    # Reuse BaseLLM sampling; restore the Reader's config even if sampling fails.
    previous = llm.llm_model.generation_config
    config = deepcopy(previous)
    config.top_k = args.sampling_top_k
    config.repetition_penalty = 1.0
    llm.llm_model.generation_config = config
    try:
        raw = llm.generate_sample(prompt, 256, temperature=args.sampling_temperature,
                                  top_p=args.sampling_top_p, seed=spec["seed"])
    finally:
        llm.llm_model.generation_config = previous
    return raw, audit
