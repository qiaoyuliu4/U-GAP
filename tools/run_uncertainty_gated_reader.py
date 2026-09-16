"""U-GAP runtime components extracted from the research implementation."""

from src.uncertainty_gate import entropy_from_scores, semantic_direct_prompt

def gate_mcq_entropy(answer, query, threshold):
    raw = answer.call_direct(query)
    decoding = dict(answer.last_decoding)
    scores = {str(key): float(value) for key, value in decoding["logprob_scores"].items()}
    probabilities, entropy = entropy_from_scores(scores)
    prediction = str(decoding["choice"])
    return {
        "uncertain": entropy >= threshold,
        "no_rag_prediction": prediction,
        "no_rag_raw_output": raw,
        "option_scores": scores,
        "option_probs": probabilities,
        "normalized_option_entropy": entropy,
        "predicted_option": prediction,
        "option_entropy_threshold": threshold,
    }

def gate_semantic_entropy(llm, question, labels, threshold, max_length):
    prompt = semantic_direct_prompt(question, labels, sampled=False)
    result = llm.score_candidate_labels(prompt, labels, max_length=max_length)
    scores = result["candidate_scores"]
    probabilities, entropy = entropy_from_scores(scores)
    prediction = result["choice"]
    return {
        "uncertain": entropy >= threshold,
        "no_rag_prediction": prediction,
        "no_rag_raw_output": None,
        "option_scores": scores,
        "option_probs": probabilities,
        "normalized_option_entropy": entropy,
        "predicted_option": prediction,
        "option_entropy_threshold": threshold,
        "candidate_token_logprobs": result["candidate_token_logprobs"],
        "score_normalization": result["score_normalization"],
    }
