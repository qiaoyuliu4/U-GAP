"""U-GAP runtime components extracted from the research implementation."""

import math

from src.promptTemplate import answer_generation_direct_prompt_template

DATASET_ALIASES = {
    "medqa": "medqa",
    "medmcqa": "medmcqa",
    "mmlu": "mmlu",
    "mmlu_med": "mmlu",
    "mmlu-med": "mmlu",
    "medddx": "MedDDx",
    "pubmedqa": "pubmedqa",
    "bioasq": "bioasq",
}

SEMANTIC_LABEL_DATASETS = {"pubmedqa", "bioasq"}

def canonical_dataset_name(name):
    key = str(name or "").strip().lower()
    if key not in DATASET_ALIASES:
        raise ValueError("Unsupported uncertainty-gate dataset: %s" % name)
    return DATASET_ALIASES[key]

def normalized_dataset_key(name):
    return canonical_dataset_name(name).lower()

def reader_query(example):
    lines = [str(example["question"]).strip()]
    lines.extend(
        "%s: %s" % (label, value)
        for label, value in example["options"].items()
    )
    return "\n".join(lines)

def candidate_space(example, dataset):
    options = {str(key).upper(): str(value).strip() for key, value in example["options"].items()}
    if len(options) < 2:
        raise ValueError("At least two answer candidates are required at index %s" % example["index"])
    if normalized_dataset_key(dataset) in SEMANTIC_LABEL_DATASETS:
        labels = [value.lower() for value in options.values()]
        if len(set(labels)) != len(labels):
            raise ValueError("Semantic answer labels must be unique: %s" % labels)
        option_to_candidate = dict(zip(options, labels))
    else:
        labels = list(options)
        option_to_candidate = {label: label for label in labels}
    candidate_to_option = {value: key for key, value in option_to_candidate.items()}
    return {
        "labels": labels,
        "options": options,
        "option_to_candidate": option_to_candidate,
        "candidate_to_option": candidate_to_option,
    }

def canonical_answer(example, space):
    answer = str(example.get("answer", "")).strip()
    upper = answer.upper()
    if upper in space["option_to_candidate"]:
        return space["option_to_candidate"][upper]
    folded = {label.casefold(): label for label in space["labels"]}
    if answer.casefold() in folded:
        return folded[answer.casefold()]
    return answer

def entropy_from_scores(scores):
    if len(scores) < 2:
        raise ValueError("Normalized entropy requires at least two candidate scores")
    values = [float(value) for value in scores.values()]
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    probabilities = [weight / total for weight in weights]
    entropy = -sum(probability * math.log(probability) for probability in probabilities if probability > 0)
    normalized = entropy / math.log(len(probabilities))
    normalized = min(1.0, max(0.0, normalized))
    return dict(zip(scores, probabilities)), normalized

def semantic_direct_prompt(question, labels, sampled=False):
    allowed = ", ".join(labels)
    instruction = (
        "Available answer labels:\n%s\n\n"
        "This is a biomedical question. Select exactly one answer label from: %s."
    ) % (allowed, allowed)
    prompt = (
        answer_generation_direct_prompt_template
        .replace("{answer_instruction}", instruction)
        .replace("{q}", str(question).strip())
    ).rstrip()
    if sampled:
        prompt += (
            "\n\nReturn exactly one JSON object with the key \"Answer\" and one of "
            "%s as its value. Do not provide an explanation." % allowed
        )
    else:
        prompt += (
            "\n\nChoose exactly one label from: %s.\n"
            "Return only the final answer in this format:\n[ANSWER] <label> [SOLVED]" % allowed
        )
    return prompt
