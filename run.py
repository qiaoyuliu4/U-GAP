"""Public configuration entry point for U-GAP, independent of historical caches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PATH_KEYS = {"benchmark_path", "llama_model", "qwen_model", "ce_model", "query_model",
             "article_model", "medrag_repo", "db_dir", "retrieval_python", "output_dir"}


def read_config(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise ValueError("Install requirements.txt to read YAML configurations") from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a mapping")
    return value


def resolve_path(value):
    expanded = os.path.expandvars(str(value))
    if "$" in expanded:
        raise ValueError("Unresolved environment variable in path: " + expanded)
    path = Path(expanded).expanduser()
    return str((ROOT / path).resolve() if not path.is_absolute() else path.resolve())


def validate_questions(rows, dataset):
    from src.uncertainty_gate import candidate_space, canonical_answer
    result = {}
    for index, row in enumerate(rows):
        key = str(row.get("id", row.get("source_key", "")))
        if not key or key in result:
            raise ValueError("Each question requires a nonempty unique id/source_key")
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError("Missing question: " + key)
        options = row.get("options")
        if not isinstance(options, dict) or not 2 <= len(options) <= 5:
            raise ValueError("options must map 2-5 labels to text: " + key)
        if any(k not in "ABCDE" or len(k) != 1 or not isinstance(v, str) or not v.strip()
               for k, v in options.items()):
            raise ValueError("Options require uppercase A-E labels and nonempty string values")
        if dataset in {"pubmedqa", "bioasq"}:
            expected = {"A": "yes", "B": "no", **({"C": "maybe"} if dataset == "pubmedqa" else {})}
            if {k: v.lower().strip() for k, v in options.items()} != expected:
                raise ValueError("Semantic task option mapping must be " + str(expected))
        example = {"index": index, "question": row["question"], "options": options,
                   "answer": row.get("answer", "")}
        space = candidate_space(example, dataset)
        if example["answer"] and canonical_answer(example, space) not in space["labels"]:
            raise ValueError("Invalid gold answer: " + key)
        result[key] = {k: example[k] for k in ["question", "options", "answer"]}
    if not result:
        raise ValueError("No input questions")
    return {dataset: result}


def prepare_input(config, questions=None, limit=-1, write=True):
    dataset = "mmlu" if config["dataset"] == "mmlu_med" else config["dataset"]
    if dataset not in {"medqa", "mmlu", "pubmedqa", "bioasq"}:
        raise ValueError("Supported datasets: medqa, mmlu, mmlu_med, pubmedqa, bioasq")
    if limit == 0 or limit < -1:
        raise ValueError("limit must be positive or -1")
    if questions:
        path = Path(resolve_path(questions))
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    else:
        raw = json.loads(Path(config["benchmark_path"]).read_text(encoding="utf-8-sig"))
        if dataset not in raw:
            raise ValueError("Dataset key absent from benchmark: " + dataset)
        rows = [{**raw[dataset][key], "id": key} for key in sorted(raw[dataset])]
    # Validate the complete input before taking a subset, including duplicate IDs.
    validated = validate_questions(rows, dataset)
    if limit > 0:
        keys = sorted(validated[dataset])[:limit]
        validated[dataset] = {k: validated[dataset][k] for k in keys}
    count = len(validated[dataset])
    expected = int(config.get("expected_total", -1))
    if expected > 0 and expected != count:
        raise ValueError("expected_total=%d but selected input has %d records" % (expected, count))
    config["expected_total"] = count
    if questions or limit > 0:
        payload = json.dumps(validated, ensure_ascii=False, sort_keys=True, indent=2)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        destination = Path(config["output_dir"]).parent / ".ugap_inputs" / (digest + ".json")
        if write:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and destination.read_text(encoding="utf-8") != payload:
                raise ValueError("Prepared input collision")
            if not destination.exists():
                destination.write_text(payload, encoding="utf-8")
        config["benchmark_path"] = str(destination)
    return count


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    p.add_argument("--stage", choices=["all", "audit", "gate", "rag", "reader", "g5", "fusion", "finalize", "prepare_indexes"], default="all")
    p.add_argument("--dataset", choices=["medqa", "mmlu", "mmlu_med", "pubmedqa", "bioasq"])
    p.add_argument("--output-dir")
    p.add_argument("--questions", help="JSONL questions; replaces benchmark_path")
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--check", action="store_true", help="Validate configuration/input without models, asset access or writes")
    args = p.parse_args(argv)
    from tools.run_final_method_end_to_end import parser as method_parser
    config = read_config(resolve_path(args.config))
    for item in args.set:
        key, separator, value = item.partition("=")
        if not separator:
            p.error("--set requires KEY=VALUE")
        try:
            config[key] = json.loads(value)
        except json.JSONDecodeError:
            config[key] = value
    if args.dataset:
        config["dataset"] = args.dataset
    if args.output_dir:
        config["output_dir"] = args.output_dir
    allowed = {a.dest for a in method_parser()._actions} - {"help", "stage"}
    if set(config) - allowed:
        p.error("Unknown config keys: " + ", ".join(sorted(set(config) - allowed)))
    for key in PATH_KEYS:
        if config.get(key):
            config[key] = resolve_path(config[key])
    if not config.get("retrieval_python"):
        config["retrieval_python"] = sys.executable
    if args.stage != "prepare_indexes":
        count = prepare_input(config, args.questions, args.limit, write=not args.check)
        print("U-GAP dataset=%s selected records=%d" % (config["dataset"], count), flush=True)
    command = []
    for key, value in config.items():
        command.extend(["--" + key, str(value)])
    parsed = method_parser().parse_args(command)
    if not 0 <= parsed.option_entropy_threshold <= 1 or parsed.generation_documents < 1:
        p.error("Threshold must be in [0,1]; generation_documents must be positive")
    if args.check:
        print(json.dumps(config, indent=2))
        print("Configuration/input check passed. No model inference executed.")
        return
    if args.stage == "prepare_indexes":
        command = [config["retrieval_python"], str(ROOT / "tools/run_evidence_gap_rag.py"),
                   "--stage", "prepare_indexes", "--run_dir", config["output_dir"]]
        for key in ["medrag_repo", "db_dir", "corpus", "query_model", "article_model"]:
            if config.get(key):
                command += ["--" + key, str(config[key])]
    else:
        command = [sys.executable, str(ROOT / "tools/run_final_method_end_to_end.py"),
                   *command, "--stage", args.stage]
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
