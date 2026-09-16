"""U-GAP runtime components extracted from the research implementation."""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

from collections import Counter

import json

import math

import os

from pathlib import Path

import subprocess

import sys

import time

from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]

from src.answer_decoding import parse_mcq_prediction

from src.conflict_aware_generation_fusion import build_direct_conflict_aware_fusion_prompt

from src.evidence_analysis_reader import FinalEvidenceLookup

from src.parametric_generation_multidoc import CheckedReaderLLM, document_specs, generation_prompt, render_documents, sample_document

from src.parametric_generation_second_pass import build_generation_only_reader_prompt, clean_generated_background, extract_final_generation_information_need, is_target_record

from src.uncertainty_gate import SEMANTIC_LABEL_DATASETS, candidate_space, canonical_answer, canonical_dataset_name, normalized_dataset_key, reader_query

from tools.run_final_prompt_control import call_reader

from tools.run_parametric_generation_second_pass import append_jsonl, file_sha256, index_rows, read_jsonl, record_sha256, write_json_atomic

from tools.run_uncertainty_gated_reader import gate_mcq_entropy, gate_semantic_entropy

SCHEMA = "final_method_end_to_end_v1"

FILES = {
    "gate": "01_gate_predictions.jsonl",
    "source_keys": "02_uncertain_source_keys.json",
    "reader": "03_final_evidence_reader_predictions.jsonl",
    "generated": "04_g5_generated_backgrounds.jsonl",
    "g5": "05_g5_predictions.jsonl",
    "fusion": "06_conflict_fusion_predictions.jsonl",
    "final": "07_final_predictions.jsonl",
    "summary": "08_summary.json",
    "skipped": "09_g5_skipped_missing_gap.jsonl",
}

RAG_DIR = "rag_uncertain"

RAG_FINAL = "08_final_selected.jsonl"

MANIFEST = "00_manifest.json"

RUN_LOG = "run.log"

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["audit", "gate", "rag", "reader", "g5", "fusion", "finalize", "all"],
                   default="all")
    p.add_argument("--dataset", required=True,
                   choices=["medqa", "medmcqa", "mmlu", "mmlu_med", "medddx", "MedDDx",
                            "pubmedqa", "bioasq"])
    p.add_argument("--benchmark_path", required=True)
    p.add_argument("--llama_model", required=True)
    p.add_argument("--qwen_model", required=True)
    p.add_argument("--ce_model", required=True)
    p.add_argument("--query_model", required=True)
    p.add_argument("--article_model", default="")
    p.add_argument("--medrag_repo", required=True)
    p.add_argument("--db_dir", required=True)
    p.add_argument("--corpus", choices=["MedText", "Textbooks", "StatPearls"], default="MedText")
    p.add_argument("--retrieval_python", default="")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--option_entropy_threshold", type=float, default=0.02)
    p.add_argument("--answer_logprob_max_length", type=int, default=8192)
    p.add_argument("--answer_tokens", type=int, default=96)
    p.add_argument("--context_limit", type=int, default=8192)
    p.add_argument("--expected_total", type=int, default=-1)
    p.add_argument("--num_views", type=int, choices=[2, 3], default=3)
    p.add_argument("--per_query_k", type=int, default=8)
    p.add_argument("--rrf_k", type=int, default=100)
    p.add_argument("--pool_size", type=int, default=32)
    p.add_argument("--min_selected", type=int, default=4)
    p.add_argument("--max_selected", type=int, default=8)
    p.add_argument("--context_chars", type=int, default=8000)
    p.add_argument("--evidence_chars", type=int, default=1200)
    p.add_argument("--pool_batch", type=int, default=6)
    p.add_argument("--pool_check_backend", choices=["qwen", "cross_encoder"], default="cross_encoder")
    p.add_argument("--pool_ce_topk_per_gap", type=int, default=3)
    p.add_argument("--plan_tokens", type=int, default=700)
    p.add_argument("--esa_tokens", type=int, default=1600)
    p.add_argument("--qwen_context", type=int, default=8192)
    p.add_argument("--ce_batch_size", type=int, default=8)
    p.add_argument("--selection_stop_gain", type=float, default=0.60)
    p.add_argument("--near_duplicate_threshold", type=float, default=0.85)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--g5_temperature", type=float, default=1.2)
    p.add_argument("--g5_top_p", type=float, default=0.9)
    p.add_argument("--g5_top_k", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--generation_documents", type=int, default=5)
    return p

def log_line(output, message):
    print(message, flush=True)
    with (output / RUN_LOG).open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")

def load_examples(args):
    from tools.export_medrag_snippets import load_kgarevion_dataset
    dataset = canonical_dataset_name(args.dataset)
    examples = load_kgarevion_dataset(dataset, args.benchmark_path)
    if not examples:
        raise ValueError("Empty dataset")
    if args.expected_total > 0 and len(examples) != args.expected_total:
        raise ValueError("Expected %d records, got %d" % (args.expected_total, len(examples)))
    by_index = {int(row["index"]): row for row in examples}
    if len(by_index) != len(examples):
        raise ValueError("Duplicate dataset indexes")
    keys = [str(row["source_key"]) for row in examples]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate dataset source keys")
    return dataset, examples, by_index

def model_metadata(path):
    root = Path(path)
    if not (root / "config.json").is_file():
        raise FileNotFoundError(root / "config.json")
    return {name: file_sha256(root / name) for name in
            ["config.json", "generation_config.json", "tokenizer_config.json",
             "tokenizer.json", "model.safetensors.index.json"] if (root / name).is_file()}

def build_manifest(args):
    code = sorted(p for folder in ["src", "tools", "action"] for p in (ROOT / folder).glob("*.py"))
    code += [ROOT / "run.py"]
    config = vars(args).copy()
    config.pop("stage")
    config.pop("output_dir")
    for name in ["benchmark_path", "llama_model", "qwen_model", "ce_model", "query_model",
                 "article_model", "medrag_repo", "db_dir", "retrieval_python"]:
        if config.get(name):
            config[name] = str(Path(config[name]).resolve())
    return {
        "schema": SCHEMA,
        "config": config,
        "benchmark_sha256": file_sha256(args.benchmark_path),
        "code_sha256": {str(path.resolve()): file_sha256(path) for path in code},
        "llama_metadata": model_metadata(args.llama_model),
        "method": {
            "gate": "candidate-label normalized entropy; uncertain if entropy >= %s" % args.option_entropy_threshold,
            "rag_scope": "fresh Evidence-Gap RAG only for uncertain source keys",
            "reader": "Direct instructions plus freshly constructed Final evidence",
            "g5_scope": "fresh %d-document generation for second-pass-insufficient only" % args.generation_documents,
            "fusion": "Final-C/G5 agreement keep; disagreement uses conflict-aware Direct prompt",
            "cached_experimental_evidence": False,
        },
    }

def ensure_output(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    expected = build_manifest(args)
    path = output / MANIFEST
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8-sig")) != expected:
            raise ValueError("Input/config/code changed. Use a NEW --output_dir")
    elif any(output.iterdir()):
        raise ValueError("Non-empty output_dir has no manifest. Use a NEW --output_dir")
    else:
        write_json_atomic(path, expected)
    return output

def validate_assets(args):
    for name in ["benchmark_path"]:
        if not Path(getattr(args, name)).is_file():
            raise FileNotFoundError(getattr(args, name))
    for name in ["llama_model", "qwen_model", "ce_model", "query_model"]:
        if not (Path(getattr(args, name)) / "config.json").is_file():
            raise FileNotFoundError(Path(getattr(args, name)) / "config.json")
    if not (Path(args.medrag_repo) / "src/utils.py").is_file():
        raise FileNotFoundError(Path(args.medrag_repo) / "src/utils.py")

def load_indexed(path, name):
    return index_rows(read_jsonl(path), name) if Path(path).is_file() else {}

def validate_identity(row, example, name):
    if int(row.get("index", -1)) != int(example["index"]):
        raise ValueError("%s index mismatch at %s" % (name, example["index"]))
    if str(row.get("source_key", "")) != str(example["source_key"]):
        raise ValueError("%s source_key mismatch at %s" % (name, example["index"]))
    if row.get("question") != example["question"] or row.get("options") != example["options"]:
        raise ValueError("%s question/options mismatch at %s" % (name, example["index"]))

def exact_mcnemar(gain, harm):
    n = int(gain) + int(harm)
    if not n:
        return 1.0
    low = min(int(gain), int(harm))
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(low + 1)) / (2 ** n))

def transition(ids, before, after, gold):
    gain = sum(before[i] != gold[i] and after[i] == gold[i] for i in ids)
    harm = sum(before[i] == gold[i] and after[i] != gold[i] for i in ids)
    both = sum(before[i] == gold[i] and after[i] == gold[i] for i in ids)
    return {"records": len(ids), "wrong_to_right": gain, "right_to_wrong": harm,
            "both_correct": both, "both_wrong": len(ids) - gain - harm - both,
            "net_correct_gain": gain - harm,
            "accuracy_delta_pp": 100 * (gain - harm) / len(ids) if ids else 0.0,
            "mcnemar_exact_p": exact_mcnemar(gain, harm)}

class GateLogger:
    def __init__(self, delegate):
        self.delegate = delegate
        self.llm_name = delegate.llm_name
        self.last_prompt = None

    def score_choices(self, prompt, choices, max_length=8192):
        self.last_prompt = prompt
        return self.delegate.score_choices(prompt, choices, max_length=max_length)

    def score_candidate_labels(self, prompt, labels, max_length=8192):
        self.last_prompt = prompt
        return self.delegate.score_candidate_labels(prompt, labels, max_length=max_length)

def load_llama(args, gate=False):
    os.environ["LLAMA3_MODEL_PATH"] = str(Path(args.llama_model).resolve())
    from action.answer import Answer
    from src.utils import BaseLLM
    base = BaseLLM("llama3")
    if gate:
        logged = GateLogger(base)
        answer = Answer(logged, SimpleNamespace(
            answer_decoding="choice_logprob", answer_tokens=args.answer_tokens,
            answer_logprob_max_length=args.answer_logprob_max_length,
            selected_evidence_prompt_style="structured"))
        return logged, answer, base
    checked = CheckedReaderLLM(base, args.context_limit)
    answer = Answer(checked, SimpleNamespace(
        answer_decoding="generate", answer_tokens=args.answer_tokens,
        selected_evidence_prompt_style="structured"))
    return checked, answer, base

def run_gate(args, output, dataset, examples):
    path = output / FILES["gate"]
    rows = load_indexed(path, FILES["gate"])
    for i, row in rows.items():
        validate_identity(row, examples[i], "Gate cache")
    if len(rows) == len(examples):
        log_line(output, "Gate cached: %d" % len(rows))
    else:
        logged, answer, base = load_llama(args, gate=True)
        semantic = normalized_dataset_key(dataset) in SEMANTIC_LABEL_DATASETS
        for position, example in enumerate(examples, 1):
            i = int(example["index"])
            if i in rows:
                continue
            space = candidate_space(example, dataset)
            query = reader_query(example)
            if semantic:
                gate = gate_semantic_entropy(logged, example["question"], space["labels"],
                                             args.option_entropy_threshold,
                                             args.answer_logprob_max_length)
                gate_prompt = logged.last_prompt
            else:
                gate = gate_mcq_entropy(answer, query, args.option_entropy_threshold)
                gate_prompt = logged.last_prompt
            row = {"index": i, "source_key": str(example["source_key"]), "dataset": dataset,
                   "question": example["question"], "options": example["options"],
                   "answer": canonical_answer(example, space), "candidate_labels": space["labels"],
                   "gate_type": "option_entropy", "gate_prompt": gate_prompt, **gate}
            row["correct"] = row["no_rag_prediction"] == row["answer"]
            append_jsonl(path, row)
            rows[i] = row
            log_line(output, "gate [%d/%d] index=%d uncertain=%s" %
                     (position, len(examples), i, row["uncertain"]))
    uncertain = [str(example["source_key"]) for example in examples if rows[int(example["index"])]["uncertain"]]
    write_json_atomic(output / FILES["source_keys"], {"source_keys": uncertain})
    log_line(output, "Gate complete: total=%d confident=%d uncertain=%d" %
             (len(examples), len(examples) - len(uncertain), len(uncertain)))

def rag_command(args, output, dataset):
    command = [sys.executable, str(ROOT / "tools/run_evidence_gap_rag.py"), "--stage", "all",
               "--run_dir", str(output / RAG_DIR), "--dataset", dataset,
               "--benchmark_path", args.benchmark_path,
               "--source_keys_path", str(output / FILES["source_keys"]),
               "--qwen_model", args.qwen_model, "--ce_model", args.ce_model,
               "--query_model", args.query_model, "--medrag_repo", args.medrag_repo,
               "--db_dir", args.db_dir, "--corpus", args.corpus,
               "--num_views", str(args.num_views), "--per_query_k", str(args.per_query_k),
               "--rrf_k", str(args.rrf_k), "--pool_size", str(args.pool_size),
               "--min_selected", str(args.min_selected), "--max_selected", str(args.max_selected),
               "--context_chars", str(args.context_chars), "--evidence_chars", str(args.evidence_chars),
               "--pool_batch", str(args.pool_batch), "--pool_check_backend", args.pool_check_backend,
               "--pool_ce_topk_per_gap", str(args.pool_ce_topk_per_gap),
               "--plan_tokens", str(args.plan_tokens), "--esa_tokens", str(args.esa_tokens),
               "--qwen_context", str(args.qwen_context), "--ce_batch_size", str(args.ce_batch_size),
               "--selection_stop_gain", str(args.selection_stop_gain),
               "--near_duplicate_threshold", str(args.near_duplicate_threshold),
               "--device", args.device, "--max_samples", "-1"]
    if args.article_model:
        command.extend(["--article_model", args.article_model])
    if args.retrieval_python:
        command.extend(["--retrieval_python", args.retrieval_python])
    return command

def load_final_rows(output, examples, uncertain):
    path = output / RAG_DIR / RAG_FINAL
    rows = load_indexed(path, "fresh Final evidence")
    if set(rows) != set(uncertain):
        raise ValueError("Fresh Final evidence must contain exactly the uncertain indexes")
    for i, row in rows.items():
        validate_identity(row, examples[i], "Fresh Final evidence")
    return rows

def run_reader(args, output, dataset, examples, uncertain, final_rows):
    path = output / FILES["reader"]
    cached = load_indexed(path, FILES["reader"])
    if not set(cached).issubset(set(uncertain)):
        raise ValueError("Reader cache contains confident records")
    for i, row in cached.items():
        validate_identity(row, examples[i], "Final-C Reader cache")
        if row.get("selected_evidence") != final_rows[i].get("selected_evidence"):
            raise ValueError("Final-C Reader evidence changed at index=%d" % i)
    if len(cached) == len(uncertain):
        log_line(output, "Final-C Reader cached: %d" % len(cached))
        return
    checked, answer, _ = load_llama(args)
    lookup = FinalEvidenceLookup(output / RAG_DIR / RAG_FINAL, dataset, args.evidence_chars)
    for position, i in enumerate(sorted(uncertain), 1):
        if i in cached:
            continue
        example = examples[i]
        record, alignment = lookup.find(example)
        if record is None:
            raise ValueError("Fresh Final evidence missing at index=%d: %s" % (i, alignment))
        formatted = lookup.format_record(record)
        query = reader_query(example)
        raw = call_reader(answer, query, formatted, "direct")
        space = candidate_space(example, dataset)
        option = parse_mcq_prediction(raw)
        prediction = space["option_to_candidate"].get(option)
        row = {"index": i, "source_key": str(example["source_key"]), "dataset": dataset,
               "question": example["question"], "options": example["options"],
               "answer": canonical_answer(example, space), "stage": "final_c_reader",
               "route": record.get("route"), "stop_reason": record.get("stop_reason"),
               "final_esa_status": (record.get("evidence_state") or {}).get("status"),
               "selected_evidence": record.get("selected_evidence"),
               "selected_text": formatted["selected_text"], "evidence_alignment": alignment,
               "reader_prompt": checked.last_prompt, "raw_output": raw,
               "option_prediction": option, "prediction": prediction,
               "correct": prediction == canonical_answer(example, space),
               "input_audit": checked.input_audit}
        append_jsonl(path, row)
        cached[i] = row
        log_line(output, "Final-C [%d/%d] index=%d prediction=%s" %
                 (position, len(uncertain), i, prediction))

def generation_tasks(final_rows, examples, seed, document_count=5):
    tasks, prepared, skipped = {}, {}, []
    for i, row in sorted(final_rows.items()):
        if not is_target_record(row):
            continue
        need = extract_final_generation_information_need(row)
        if need is None:
            skipped.append({"index": i, "source_key": str(examples[i]["source_key"]),
                            "route": row.get("route"), "stop_reason": row.get("stop_reason"),
                            "reason": "missing_final_gap"})
            continue
        base = {"index": i, "source_key": str(examples[i]["source_key"]),
                "question": examples[i]["question"], "options": examples[i]["options"],
                "route": row.get("route"), "stop_reason": row.get("stop_reason"), **need}
        prepared[i] = base
        for spec in document_specs(base, seed, document_count):
            tasks[i, spec["document_number"]] = {
                "index": i, "source_key": base["source_key"], "question": base["question"],
                "route": base["route"], "stop_reason": base["stop_reason"], **spec,
                "generator_prompt": generation_prompt(base["question"], spec)}
    return tasks, prepared, skipped

def load_generated(path, tasks):
    rows = {}
    for row in read_jsonl(path) if Path(path).is_file() else []:
        key = (int(row["index"]), int(row["document_number"]))
        if key in rows or key not in tasks:
            raise ValueError("Unexpected/duplicate generated document: %s" % (key,))
        if row.get("stage_input_sha256") != record_sha256(tasks[key]):
            raise ValueError("Generated document input changed: %s" % (key,))
        if row.get("content_sha256") != record_sha256(row.get("generated_background")):
            raise ValueError("Generated document content checksum failed: %s" % (key,))
        rows[key] = row
    return rows

def run_g5(args, output, dataset, examples, final_rows):
    tasks, prepared, skipped = generation_tasks(final_rows, examples, args.seed, args.generation_documents)
    skip_path = output / FILES["skipped"]
    if skip_path.is_file():
        if read_jsonl(skip_path) != skipped:
            raise ValueError("G5 skipped-record cache mismatch")
    else:
        for row in skipped:
            append_jsonl(skip_path, row)
    generated = load_generated(output / FILES["generated"], tasks)
    predictions = load_indexed(output / FILES["g5"], FILES["g5"])
    if not set(predictions).issubset(set(prepared)):
        raise ValueError("G5 prediction cache contains non-target records")
    for i, row in predictions.items():
        validate_identity(row, examples[i], "G5 prediction cache")
        if row.get("document_ids") != ["G%d" % n for n in range(1, args.generation_documents + 1)]:
            raise ValueError("Incomplete G5 prediction at index=%d" % i)
    if len(generated) == len(tasks) and len(predictions) == len(prepared):
        log_line(output, "G5 cached: target=%d skipped_gap=%d" % (len(prepared), len(skipped)))
        return
    checked, reader, base = load_llama(args)
    sample_args = SimpleNamespace(sampling_temperature=args.g5_temperature,
                                  sampling_top_p=args.g5_top_p,
                                  sampling_top_k=args.g5_top_k,
                                  context_limit=args.context_limit)
    for position, (key, task) in enumerate(tasks.items(), 1):
        if key in generated:
            continue
        started = time.monotonic()
        raw, audit = sample_document(base, task["generator_prompt"], task, sample_args)
        background = clean_generated_background(raw)
        if not background:
            raise ValueError("Empty G5 document at %s" % (key,))
        row = {**task, "generated_background": background, "generator_raw_output": raw,
               "input_audit": audit,
               "generator_metadata": {"model_path": str(Path(args.llama_model).resolve()),
                   "do_sample": True, "temperature": args.g5_temperature,
                   "top_p": args.g5_top_p, "top_k": args.g5_top_k,
                   "max_new_tokens": 256, "seed": task["seed"]},
               "stage_input_sha256": record_sha256(task),
               "content_sha256": record_sha256(background),
               "generation_seconds": time.monotonic() - started}
        append_jsonl(output / FILES["generated"], row)
        generated[key] = row
        log_line(output, "G5 generate [%d/%d] index=%d document=G%d" %
                 (position, len(tasks), key[0], key[1]))
    for position, (i, task) in enumerate(sorted(prepared.items()), 1):
        if i in predictions:
            continue
        documents = [generated[i, number] for number in range(1, args.generation_documents + 1)]
        background = render_documents(documents)
        query = reader_query(examples[i])
        prompt = build_generation_only_reader_prompt(reader, query, background)
        started = time.monotonic()
        raw = reader._decode(prompt, query, new_tokens_num=args.answer_tokens)
        space = candidate_space(examples[i], dataset)
        option = parse_mcq_prediction(raw)
        prediction = space["option_to_candidate"].get(option)
        gold = canonical_answer(examples[i], space)
        row = {**task, "dataset": dataset, "stage": "score_g5",
               "document_ids": ["G%d" % n for n in range(1, args.generation_documents + 1)],
               "generated_background": background, "reader_prompt": checked.last_prompt,
               "raw_output": raw, "option_prediction": option, "prediction": prediction,
               "answer": gold, "correct": prediction == gold,
               "reader_metadata": {"model_path": str(Path(args.llama_model).resolve()),
                   "decoding": "generate", "do_sample": False, "max_new_tokens": args.answer_tokens},
               "input_audit": checked.input_audit, "scoring_seconds": time.monotonic() - started}
        append_jsonl(output / FILES["g5"], row)
        predictions[i] = row
        log_line(output, "G5 score [%d/%d] index=%d prediction=%s" %
                 (position, len(prepared), i, prediction))
    log_line(output, "G5 complete: second_pass_insufficient=%d generated=%d skipped_gap=%d" %
             (len(prepared) + len(skipped), len(prepared), len(skipped)))

def validate_generation_complete(args, output, examples, final_rows):
    tasks, prepared, skipped = generation_tasks(final_rows, examples, args.seed, args.generation_documents)
    if set(load_generated(output / FILES["generated"], tasks)) != set(tasks):
        raise ValueError("Generated backgrounds are incomplete; run --stage g5 first")
    predictions = load_indexed(output / FILES["g5"], FILES["g5"])
    if set(predictions) != set(prepared):
        raise ValueError("G5 predictions are incomplete; run --stage g5 first")
    skip_path = output / FILES["skipped"]
    if (read_jsonl(skip_path) if skip_path.is_file() else []) != skipped:
        raise ValueError("Missing-gap records are incomplete; run --stage g5 first")


def run_fusion(args, output, dataset, examples, final_rows):
    validate_generation_complete(args, output, examples, final_rows)
    readers = load_indexed(output / FILES["reader"], FILES["reader"])
    g5 = load_indexed(output / FILES["g5"], FILES["g5"])
    disagreements = set()
    for i, row in g5.items():
        valid = set(candidate_space(examples[i], dataset)["labels"])
        final_c = readers[i].get("prediction")
        g5_prediction = row.get("prediction")
        if not (final_c == g5_prediction and final_c in valid):
            disagreements.add(i)
    cached = load_indexed(output / FILES["fusion"], FILES["fusion"])
    if not set(cached).issubset(disagreements):
        raise ValueError("Fusion cache contains agreement/non-target records")
    for i, row in cached.items():
        validate_identity(row, examples[i], "Fusion cache")
    if len(cached) == len(disagreements):
        log_line(output, "Fusion cached: disagreements=%d" % len(disagreements))
        return
    checked, reader, _ = load_llama(args)
    lookup = FinalEvidenceLookup(output / RAG_DIR / RAG_FINAL, dataset, args.evidence_chars)
    for position, i in enumerate(sorted(disagreements), 1):
        if i in cached:
            continue
        example = examples[i]
        record, alignment = lookup.find(example)
        if record is None:
            raise ValueError("Fusion Final evidence missing at index=%d: %s" % (i, alignment))
        formatted = lookup.format_record(record)
        query = reader_query(example)
        prompt = build_direct_conflict_aware_fusion_prompt(
            reader, query, formatted["selected_text"],
            g5[i].get("generation_information_need"), g5[i].get("generated_background"))
        started = time.monotonic()
        raw = reader._decode(prompt, query, new_tokens_num=args.answer_tokens)
        space = candidate_space(example, dataset)
        option = parse_mcq_prediction(raw)
        prediction = space["option_to_candidate"].get(option)
        gold = canonical_answer(example, space)
        row = {"index": i, "source_key": str(example["source_key"]), "dataset": dataset,
               "question": example["question"], "options": example["options"], "answer": gold,
               "stage": "conflict_fusion", "route": final_rows[i].get("route"),
               "stop_reason": final_rows[i].get("stop_reason"),
               "final_c_prediction": readers[i].get("prediction"),
               "g5_prediction": g5[i].get("prediction"),
               "selected_evidence": final_rows[i].get("selected_evidence"),
               "generation_information_need": g5[i].get("generation_information_need"),
               "generated_background": g5[i].get("generated_background"),
               "fusion_prompt": checked.last_prompt, "raw_output": raw,
               "option_prediction": option, "fusion_prediction": prediction,
               "correct": prediction == gold, "input_audit": checked.input_audit,
               "scoring_seconds": time.monotonic() - started}
        append_jsonl(output / FILES["fusion"], row)
        cached[i] = row
        log_line(output, "fusion [%d/%d] index=%d prediction=%s" %
                 (position, len(disagreements), i, prediction))

def run_finalize(args, output, dataset, examples, final_rows):
    validate_generation_complete(args, output, examples, final_rows)
    gate = load_indexed(output / FILES["gate"], FILES["gate"])
    reader = load_indexed(output / FILES["reader"], FILES["reader"])
    g5 = load_indexed(output / FILES["g5"], FILES["g5"])
    fusion = load_indexed(output / FILES["fusion"], FILES["fusion"])
    uncertain = {i for i, row in gate.items() if row["uncertain"]}
    if set(reader) != uncertain:
        raise ValueError("Final-C Reader is incomplete")
    disagreements = set()
    for i, row in g5.items():
        valid = set(candidate_space(examples[i], dataset)["labels"])
        if not (row.get("prediction") == reader[i].get("prediction")
                and row.get("prediction") in valid):
            disagreements.add(i)
    if set(fusion) != disagreements:
        raise ValueError("Conflict fusion is incomplete")
    rows, predictions, gold, route_counts = [], {}, {}, Counter()
    for i, example in sorted(examples.items()):
        space = candidate_space(example, dataset)
        valid = set(space["labels"])
        gold[i] = canonical_answer(example, space)
        gate_prediction = gate[i].get("no_rag_prediction")
        final_c = reader.get(i, {}).get("prediction")
        g5_prediction = g5.get(i, {}).get("prediction")
        fusion_prediction = fusion.get(i, {}).get("fusion_prediction")
        if not gate[i]["uncertain"]:
            prediction, route = gate_prediction, "confident_direct_logprob"
        elif i not in g5:
            prediction, route = final_c, "uncertain_fresh_final_c_reader"
            if i in final_rows and is_target_record(final_rows[i]):
                route = "uncertain_second_pass_missing_gap_final_c_fallback"
        elif final_c == g5_prediction and final_c in valid:
            prediction, route = final_c, "uncertain_second_pass_g5_agreement"
        else:
            prediction, route = fusion_prediction, "uncertain_second_pass_conflict_fusion"
            if prediction not in valid:
                prediction = next((candidate for candidate in [g5_prediction, final_c, gate_prediction]
                                   if candidate in valid), None)
                route += "_invalid_output_fallback"
        predictions[i] = prediction
        route_counts[route] += 1
        rows.append({"index": i, "source_key": str(example["source_key"]), "dataset": dataset,
                     "question": example["question"], "options": example["options"], "answer": gold[i],
                     "final_route": route, "gate_uncertain": gate[i]["uncertain"],
                     "normalized_option_entropy": gate[i]["normalized_option_entropy"],
                     "gate_prediction": gate_prediction, "fresh_final_c_prediction": final_c,
                     "g5_prediction": g5_prediction, "fusion_prediction": fusion_prediction,
                     "final_prediction": prediction, "correct": prediction == gold[i],
                     "evidence_gap_route": final_rows.get(i, {}).get("route"),
                     "evidence_gap_stop_reason": final_rows.get(i, {}).get("stop_reason")})
    final_path = output / FILES["final"]
    temporary = final_path.with_suffix(final_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(final_path)
    no_rag = {i: gate[i]["no_rag_prediction"] for i in examples}
    correct = sum(predictions[i] == gold[i] for i in examples)
    uncertain = [i for i in examples if gate[i]["uncertain"]]
    route_metrics = {}
    for route in sorted(route_counts):
        selected = [row for row in rows if row["final_route"] == route]
        route_correct = sum(row["correct"] for row in selected)
        route_metrics[route] = {"records": len(selected), "correct": route_correct,
                                "accuracy": route_correct / len(selected)}
    summary = {"schema": SCHEMA, "dataset": dataset, "records": len(examples),
               "correct": correct, "accuracy": correct / len(examples),
               "gate": {"threshold": args.option_entropy_threshold,
                         "confident": len(examples) - len(uncertain), "uncertain": len(uncertain),
                         "no_rag_correct": sum(no_rag[i] == gold[i] for i in examples),
                         "no_rag_accuracy": sum(no_rag[i] == gold[i] for i in examples) / len(examples)},
               "fresh_rag_records": len(final_rows),
               "second_pass_insufficient": sum(is_target_record(row) for row in final_rows.values()),
               "g5_predictions": len(g5), "fusion_predictions": len(fusion),
               "route_counts": dict(route_counts), "route_metrics": route_metrics,
               "gate_no_rag_to_final": transition(sorted(examples), no_rag, predictions, gold)}
    write_json_atomic(output / FILES["summary"], summary)
    log_line(output, "Completed end-to-end final method.")
    log_line(output, "accuracy: %d/%d = %.3f%%" % (correct, len(examples), 100 * correct / len(examples)))
    log_line(output, "routes: %s" % dict(route_counts))

def clean_stage_args(argv):
    result, skip = [], False
    for token in argv:
        if skip:
            skip = False
            continue
        if token == "--stage":
            skip = True
        elif not token.startswith("--stage="):
            result.append(token)
    return result

def main(argv=None):
    args = parser().parse_args(argv)
    if not 0 <= args.option_entropy_threshold <= 1 or args.generation_documents < 1:
        raise ValueError("Entropy threshold must be in [0, 1]; generation_documents must be positive")
    validate_assets(args)
    dataset, example_list, examples = load_examples(args)
    output = ensure_output(args)
    log_line(output, "dataset=%s records=%d output=%s" % (dataset, len(examples), output.resolve()))
    log_line(output, "fresh evidence mode: no previous 04/08/G5/prediction inputs are accepted")
    if args.stage == "audit":
        log_line(output, "Audit passed")
        return
    if args.stage == "all":
        raw = list(sys.argv[1:] if argv is None else argv)
        common = clean_stage_args(raw)
        for stage in ["gate", "rag", "reader", "g5", "fusion", "finalize"]:
            log_line(output, "Dispatch stage: " + stage)
            subprocess.run([sys.executable, str(Path(__file__).resolve()), *common, "--stage", stage], check=True)
        return
    if args.stage == "gate":
        run_gate(args, output, dataset, example_list)
        return
    gate = load_indexed(output / FILES["gate"], FILES["gate"])
    if set(gate) != set(examples):
        raise ValueError("Gate is incomplete; run --stage gate first")
    uncertain = {i for i, row in gate.items() if row["uncertain"]}
    if not uncertain:
        if args.stage == "rag":
            log_line(output, "No uncertain records; fresh RAG skipped")
            return
        final_rows = {}
    elif args.stage == "rag":
        command = rag_command(args, output, dataset)
        log_line(output, "Starting fresh Evidence-Gap RAG for %d uncertain records" % len(uncertain))
        subprocess.run(command, check=True)
        load_final_rows(output, examples, uncertain)
        log_line(output, "Fresh Evidence-Gap RAG complete")
        return
    else:
        final_rows = load_final_rows(output, examples, uncertain)
    if args.stage == "reader":
        run_reader(args, output, dataset, examples, uncertain, final_rows)
    elif args.stage == "g5":
        if set(load_indexed(output / FILES["reader"], FILES["reader"])) != uncertain:
            raise ValueError("Final-C Reader is incomplete; run --stage reader first")
        run_g5(args, output, dataset, examples, final_rows)
    elif args.stage == "fusion":
        run_fusion(args, output, dataset, examples, final_rows)
    else:
        run_finalize(args, output, dataset, examples, final_rows)

if __name__ == "__main__":
    main()
