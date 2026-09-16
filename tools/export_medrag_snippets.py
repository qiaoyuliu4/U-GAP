"""U-GAP runtime components extracted from the research implementation."""

import json

import string

def load_kgarevion_dataset(dataset_name, benchmark_path):
    if dataset_name in {"MedDDx", "MedDDx-Basic", "MedDDx-Intermediate", "MedDDx-Expert"}:
        return load_medddx_dataset(dataset_name, benchmark_path)

    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    data_key = dataset_name.lower().split("_")[0]
    if data_key not in benchmark:
        raise KeyError(f"{dataset_name} not found in {benchmark_path}")

    dataset = benchmark[data_key]
    keys = sorted(dataset.keys())
    examples = []

    for idx, key in enumerate(keys):
        item = dataset[key]
        question = item["question"]
        raw_options = item.get("options", {})

        if isinstance(raw_options, dict):
            options = {str(k).upper(): str(v) for k, v in raw_options.items()}
        else:
            labels = list(string.ascii_uppercase)
            options = {
                labels[i]: str(value)
                for i, value in enumerate(raw_options)
            }

        examples.append(
            {
                "index": idx,
                "source_key": key,
                "question": question,
                "options": options,
                "answer": item.get("answer", ""),
            }
        )

    return examples

def load_medddx_dataset(dataset_name, medddx_path):
    with open(medddx_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    buckets = {
        "MedDDx": {},
        "MedDDx-Basic": {},
        "MedDDx-Intermediate": {},
        "MedDDx-Expert": {},
    }

    for raw_idx, item in enumerate(raw_data):
        if item["sim_level_std"] > 0.04:
            split_name = "MedDDx-Basic"
        elif item["sim_level_std"] < 0.02:
            split_name = "MedDDx-Expert"
        else:
            split_name = "MedDDx-Intermediate"

        buckets["MedDDx"][raw_idx] = item
        buckets[split_name][raw_idx] = item

    if dataset_name not in buckets:
        raise KeyError(f"{dataset_name} not found in {medddx_path}")

    examples = []
    for idx, source_key in enumerate(sorted(buckets[dataset_name].keys())):
        item = buckets[dataset_name][source_key]
        question, options = parse_medddx_query(item["query"])

        examples.append(
            {
                "index": idx,
                "source_key": str(source_key),
                "question": question,
                "options": options,
                "answer": item.get("answer", ""),
            }
        )

    return examples

def parse_medddx_query(query):
    lines = [line.strip() for line in str(query).splitlines() if line.strip()]
    question_lines = []
    options = {}

    for line in lines:
        if len(line) >= 2 and line[0].upper() in string.ascii_uppercase and line[1] == ":":
            options[line[0].upper()] = line[2:].strip()
        else:
            question_lines.append(line)

    return " ".join(question_lines).strip(), options
