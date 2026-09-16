"""U-GAP runtime components extracted from the research implementation."""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

import copy

import hashlib

import json

import os

from pathlib import Path

import subprocess

import sys

import time

ROOT = Path(__file__).resolve().parents[1]

from src.evidence_gap import CORPORA, PIPELINE, apply_final_state, assert_same_question, bridge_prompt, identity, initial_queries, merge_candidates, plan_prompt, pool_prompt, select_evidence, state_prompt, unknown_state, validate_bridges, validate_plan, validate_pool_check, validate_state

STAGES = ["plan", "retrieve", "rerank", "select", "analyze", "followup", "merge_rerank", "finalize"]

FILES = dict(zip(STAGES, ["01_plans.jsonl", "02_retrieved.jsonl", "03_candidates.jsonl",
                          "04_initial_selected.jsonl", "05_analysis.jsonl", "06_followup.jsonl",
                          "07_merged_candidates.jsonl", "08_final_selected.jsonl"]))

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

def load_rows(path):
    rows, seen = [], set()
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError("Invalid JSONL %s physical line %d: %s" % (path, line_no, exc)) from exc
            key = identity(row)
            if key in seen:
                raise ValueError("Duplicate question: %s in %s" % (key, path))
            seen.add(key)
            rows.append(row)
    return rows

def load_source_keys(path):
    """Load an optional exact source-key subset without changing dataset indexes."""
    if not path:
        return None
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError("Missing --source_keys_path: " + str(source))
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        payload = payload.get("source_keys")
    if not isinstance(payload, list) or not payload:
        raise ValueError("--source_keys_path must contain a non-empty JSON list or {source_keys: [...]} object")
    keys = [str(value) for value in payload]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate source keys in --source_keys_path")
    return set(keys)

def run_cached(path, inputs, operation):
    """Never silently reuse records from changed inputs or skip damaged JSONL."""
    path = Path(path)
    partial = path.with_suffix(path.suffix + ".partial")
    existing = load_rows(path if path.exists() else partial) if path.exists() or partial.exists() else []
    if len(existing) > len(inputs):
        raise ValueError("Checkpoint has more records than current input; use a new run directory")
    for i, row in enumerate(existing):
        if row.get("stage_input_sha256") != digest(inputs[i]):
            raise ValueError("Checkpoint/input changed at index %s; use a new run directory" % row["index"])
    if path.exists():
        if len(existing) != len(inputs):
            raise ValueError("Completed file is truncated; restore the original or use a new run directory")
        print("Cached:", path, "records=", len(existing), flush=True)
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("a", encoding="utf-8", newline="\n") as handle:
        for i in range(len(existing), len(inputs)):
            start = time.monotonic()
            row = operation(inputs[i])
            row["stage_input_sha256"] = digest(inputs[i])
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            existing.append(row)
            print("%s [%d/%d] index=%s %.2fs" % (path.stem, i + 1, len(inputs), row["index"],
                                               time.monotonic() - start), flush=True)
    os.replace(partial, path)
    return existing

def base_row(row):
    return {k: copy.deepcopy(v) for k, v in row.items()
            if k not in {"snippets", "selected_evidence", "stage_input_sha256"}}

def joined(left, right):
    mapped = {identity(r): r for r in right}
    if {identity(r) for r in left} != set(mapped):
        raise ValueError("Stage files must contain exactly the same question IDs")
    pairs = []
    for row in left:
        other = mapped[identity(row)]
        assert_same_question(row, other)
        if row.get("retrieval_contract") != other.get("retrieval_contract"):
            raise ValueError("Corpus/retriever contract mismatch")
        pairs.append({"left": row, "right": other})
    return pairs

def selection(row, pool, args, gaps=None, support=None, labels=None):
    return select_evidence(row, pool, args.max_selected, args.min_selected,
                           args.context_chars, args.evidence_chars, gaps, support, labels,
                           args.selection_stop_gain, args.near_duplicate_threshold)

def judge(llm, row, selected, args, gaps=None):
    if not selected:
        return unknown_state([], "empty_evidence_pool"), {"status": "not_called"}
    result, audit = llm.ask(state_prompt(row, selected, args.evidence_chars, gaps),
                            lambda d: validate_state(d, selected, args.evidence_chars, gaps), args.esa_tokens)
    return result or unknown_state(selected, "esa_parse_or_context_failure"), audit

def gap_support_from_ce(pool_gap_scores, top_k):
    support, top_ids = {}, {}
    for gap_result in pool_gap_scores:
        gap_id = gap_result["gap_id"]
        ranked = list(gap_result.get("scores", []))[:top_k]
        top_ids[gap_id] = [item["evidence_id"] for item in ranked]
        for item in ranked:
            support.setdefault(item["evidence_id"], {})[gap_id] = float(
                item["normalized_score"]
            )
    return support, top_ids

def analyze_record_with_ce(row, initial, llm, ce, args, selected, state, audit, out):
    gaps = state["missing_relations"]
    pool_gap_scores = ce.score_gaps(row, gaps, row["snippets"])
    gap_support, top_ids = gap_support_from_ce(pool_gap_scores, args.pool_ce_topk_per_gap)
    labels = {e["evidence_id"]: e["label"] for e in state["evidence_judgments"]}
    rebuilt_selected = selection(row, row["snippets"], args, gaps, gap_support, labels)
    selected_after_rebuild = rebuilt_selected or copy.deepcopy(initial["selected_evidence"])
    out.update(
        pool_check_backend="cross_encoder",
        pool_gap_scores=pool_gap_scores,
        gap_support=gap_support,
        gap_top_evidence_ids=top_ids,
        pool_covered_gap_ids=[],
        remaining_gaps=gaps,
        pool_rebuild_attempted=True,
        pool_rebuild_succeeded=bool(rebuilt_selected),
        selected_evidence=selected_after_rebuild,
        # CE relevance is used to rebuild the evidence set, but it is not a
        # calibrated sufficiency or entailment judgment. Keep the initial ESA
        # state and defer the next ESA until after the one allowed retrieval.
        state=state,
        reanalysis_audit={
            "status": "not_called",
            "reason": "disabled_after_cross_encoder_pool_rebuild",
        },
        pool_rebuild_requires_final_esa=True,
    )

    routing_gaps = gaps
    queries, query_audit = llm.ask(
        bridge_prompt(row, routing_gaps),
        lambda d: validate_bridges(d, routing_gaps, row.get("options")),
        args.plan_tokens,
    )
    out.update(
        bridge_queries=queries or [],
        bridge_query_audit=query_audit,
        route="retrieve" if queries else "unresolved",
        stop_reason="external_retrieval_requested" if queries else "no_valid_bridge_query",
    )
    return out

def analyze_record(row, initial, llm, args, ce=None):
    assert_same_question(row, initial)
    selected = copy.deepcopy(initial["selected_evidence"])
    state, audit = judge(llm, row, selected, args)
    out = base_row(row)
    out.update(selection_config=initial.get("selection_config", {}),
               initial_state=state, state=state, initial_esa_audit=audit,
               selected_evidence=selected, bridge_queries=[], gap_support={}, pool_checks=[],
               external_followup_rounds=0, route="unresolved")
    if state["status"] == "sufficient":
        out["route"] = "initial_sufficient"
        return out
    gaps = state["missing_relations"]
    if not gaps:
        out["stop_reason"] = "no_actionable_gap"
        return out
    if args.pool_check_backend == "cross_encoder":
        if ce is None:
            raise ValueError("Cross-encoder pool checking requires a CrossEncoder instance")
        return analyze_record_with_ce(row, initial, llm, ce, args, selected, state, audit, out)
    out["pool_check_backend"] = "qwen"
    covered = set()
    pool = row["snippets"]
    # Check the whole retained Top-N pool, including initial evidence if needed.
    # Unselected evidence first makes useful in-pool recovery less expensive.
    selected_ids = {e["evidence_id"] for e in selected}
    pool = sorted(pool, key=lambda e: e["evidence_id"] in selected_ids)
    for start in range(0, len(pool), args.pool_batch):
        remaining = [g for g in gaps if g["id"] not in covered]
        if not remaining:
            break
        batch = pool[start:start + args.pool_batch]
        checks, check_audit = llm.ask(pool_prompt(row, remaining, batch, args.evidence_chars),
                                     lambda d: validate_pool_check(d, remaining, batch, args.evidence_chars),
                                     args.esa_tokens)
        out["pool_checks"].append({"candidate_ids": [e["evidence_id"] for e in batch],
                                   "checks": checks or [], "audit": check_audit})
        for check in checks or []:
            if check["status"] == "covered":
                covered.add(check["gap_id"])
            for match in check["matches"]:
                if check["status"] != "absent":
                    score = 1.0 if check["status"] == "covered" else 0.5
                    target = out["gap_support"].setdefault(match["evidence_id"], {})
                    target[check["gap_id"]] = max(score, target.get(check["gap_id"], 0))
    out["pool_covered_gap_ids"] = sorted(covered)
    out["remaining_gaps"] = [g for g in gaps if g["id"] not in covered]
    labels = {e["evidence_id"]: e["label"] for e in state["evidence_judgments"]}
    if out["gap_support"] and not out["remaining_gaps"]:
        selected = selection(row, row["snippets"], args, gaps, out["gap_support"], labels)
        out["selected_evidence"] = selected
    if not out["remaining_gaps"]:
        rebuilt, rebuilt_audit = judge(llm, row, selected, args, gaps)
        unusable = rebuilt_audit.get("status") == "fallback" or (
            rebuilt["evidence_judgments"] and all(j["label"] == "irrelevant" for j in rebuilt["evidence_judgments"]))
        if unusable:
            out.update(selected_evidence=copy.deepcopy(initial["selected_evidence"]), state=state,
                       attempted_rebuilt_state=rebuilt, reanalysis_audit=rebuilt_audit,
                       route="pool_rebuilt", stop_reason="pool_reanalysis_failed_retain_previous")
            return out
        out.update(state=rebuilt, reanalysis_audit=rebuilt_audit, route="pool_rebuilt",
                   stop_reason="pool_rebuilt_sufficient" if rebuilt["status"] == "sufficient" else
                   "pool_rebuilt_but_unresolved")
        return out
    queries, query_audit = llm.ask(bridge_prompt(row, out["remaining_gaps"]),
                                   lambda d: validate_bridges(d, out["remaining_gaps"], row.get("options")),
                                   args.plan_tokens)
    out.update(bridge_queries=queries or [], bridge_query_audit=query_audit,
               route="retrieve" if queries else "unresolved",
               stop_reason="external_retrieval_requested" if queries else "no_valid_bridge_query")
    return out

def finalize_record(analysis, merged, llm, args):
    assert_same_question(analysis, merged)
    if analysis["route"] != "retrieve":
        return apply_final_state(analysis, analysis["selected_evidence"], analysis["state"],
                                 analysis.get("stop_reason", analysis["route"]), 0)
    gaps = analysis.get("remaining_gaps", analysis["initial_state"]["missing_relations"])
    labels = {e["evidence_id"]: e["label"] for e in analysis["state"]["evidence_judgments"]}
    selected = selection(analysis, merged["snippets"], args, gaps, analysis["gap_support"], labels)
    state, audit = judge(llm, analysis, selected, args, gaps)
    unusable = audit.get("status") == "fallback" or (
        state["evidence_judgments"] and all(j["label"] == "irrelevant" for j in state["evidence_judgments"]))
    if unusable:
        stop = "second_pass_analysis_failed_retain_previous" if audit.get("status") == "fallback" else "second_pass_all_irrelevant_retain_previous"
        out = apply_final_state(analysis, analysis["selected_evidence"], analysis["state"], stop, 1)
        out.update(final_esa_audit=audit, attempted_final_state=state,
                   attempted_selected_evidence=selected, best_available_fallback=True,
                   failure_reason=stop, merged_candidate_count=len(merged["snippets"]),
                   new_unique_candidates=merged["new_unique_candidates"])
        return out
    stop = "second_pass_sufficient" if state["status"] == "sufficient" else "one_followup_limit_reached"
    out = apply_final_state(analysis, selected, state, stop, 1)
    out["final_esa_audit"] = audit
    out["merged_candidate_count"] = len(merged["snippets"])
    out["new_unique_candidates"] = merged["new_unique_candidates"]
    return out

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["all", "prepare_indexes", "audit"] + STAGES, default="all")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--dataset", default="medqa")
    p.add_argument("--benchmark_path", default="dataset/benchmark.json")
    p.add_argument("--source_keys_path", default="",
                   help="Optional JSON source-key subset. Default empty keeps the original full-dataset behavior.")
    p.add_argument("--eval_start_index", type=int, default=0)
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--qwen_model", default="models/Qwen3-8B")
    p.add_argument("--ce_model", default="models/MedCPT-Cross-Encoder")
    p.add_argument("--query_model", default="models/MedCPT-Query-Encoder")
    p.add_argument("--article_model", default="")
    p.add_argument("--medrag_repo", default="../MedRAG")
    p.add_argument("--db_dir", default="../medtext_corpus")
    p.add_argument("--corpus", choices=list(CORPORA), default="MedText")
    p.add_argument("--retrieval_python", default="", help="Optional legacy MedRAG Python, used by --stage all")
    p.add_argument("--num_views", type=int, choices=[2, 3], default=3)
    p.add_argument("--per_query_k", type=int, default=8)
    p.add_argument("--rrf_k", type=int, default=100)
    p.add_argument("--pool_size", type=int, default=32)
    p.add_argument("--max_selected", type=int, default=8)
    p.add_argument("--min_selected", type=int, default=4)
    p.add_argument("--context_chars", type=int, default=8000)
    p.add_argument("--evidence_chars", type=int, default=1200)
    p.add_argument("--pool_batch", type=int, default=6)
    p.add_argument("--pool_check_backend", choices=["qwen", "cross_encoder"], default="qwen",
                   help="Use batched Qwen checks or CE gap relevance for the retained candidate pool")
    p.add_argument("--pool_ce_topk_per_gap", type=int, default=3,
                   help="Number of highest CE-scored existing candidates mapped to each gap")
    p.add_argument("--plan_tokens", type=int, default=700)
    p.add_argument("--esa_tokens", type=int, default=1600)
    p.add_argument("--qwen_context", type=int, default=8192)
    p.add_argument("--ce_batch_size", type=int, default=8)
    p.add_argument("--selection_stop_gain", type=float, default=0.60)
    p.add_argument("--near_duplicate_threshold", type=float, default=0.85)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return p

def get_manifest(args):
    values = vars(args).copy()
    for key in ("stage", "run_dir", "retrieval_python"):
        values.pop(key)
    for key in ("benchmark_path", "source_keys_path", "qwen_model", "ce_model", "query_model", "article_model", "medrag_repo", "db_dir"):
        if values[key]:
            values[key] = str(Path(values[key]).resolve())
    values["benchmark_sha256"] = hashlib.sha256(Path(args.benchmark_path).read_bytes()).hexdigest()
    values["source_keys_sha256"] = (
        hashlib.sha256(Path(args.source_keys_path).read_bytes()).hexdigest()
        if args.source_keys_path else None
    )
    values["pipeline"] = PIPELINE
    values["implementation_sha256"] = hashlib.sha256(b"".join(
        p.read_bytes() for p in (Path(__file__), ROOT / "src/evidence_gap.py", ROOT / "src/evidence_gap_models.py")
    )).hexdigest()
    # File signatures catch changed assets without rehashing multi-GB weights on each stage.
    assets = []
    for model in (args.qwen_model, args.ce_model, args.query_model, args.article_model):
        if model:
            assets.extend(p for p in Path(model).glob("*") if p.is_file())
    for corpus in CORPORA[args.corpus]:
        root = Path(args.db_dir) / corpus
        assets.extend((root / "chunk").glob("*.jsonl"))
        assets.extend((root / "index/ncbi/MedCPT-Article-Encoder").glob("*.index"))
        assets.extend((root / "index/ncbi/MedCPT-Article-Encoder").glob("metadatas.jsonl"))
    values["asset_signature"] = digest([(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns)
                                        for p in sorted(set(assets))])
    source = Path(args.medrag_repo) / "src/utils.py"
    values["medrag_utils_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest() if source.exists() else None
    return values

def audit_run(directory):
    summary = {}
    for stage, filename in FILES.items():
        path = directory / filename
        if path.exists():
            summary[stage + "_records"] = len(load_rows(path))
    path = directory / FILES["analyze"]
    if path.exists():
        rows = load_rows(path)
        summary["routes"] = {r: sum(x["route"] == r for x in rows) for r in sorted({x["route"] for x in rows})}
        summary["external_followup_requested"] = sum(bool(x["bridge_queries"]) for x in rows)
        backends = sorted({x.get("pool_check_backend") for x in rows if x.get("pool_check_backend")})
        summary["pool_check_backends"] = {
            backend: sum(x.get("pool_check_backend") == backend for x in rows)
            for backend in backends
        }
        summary["qwen_pool_check_calls"] = sum(len(x.get("pool_checks", [])) for x in rows)
        summary["qwen_pool_reanalysis_calls"] = sum(
            bool(x.get("reanalysis_audit"))
            and x["reanalysis_audit"].get("status") != "not_called"
            for x in rows
        )
        summary["ce_pool_scored_records"] = sum(bool(x.get("pool_gap_scores")) for x in rows)
        summary["ce_pool_scored_pairs"] = sum(
            len(gap.get("scores", []))
            for x in rows
            for gap in x.get("pool_gap_scores", [])
        )
        summary["pool_rebuild_attempted"] = sum(bool(x.get("pool_rebuild_attempted")) for x in rows)
    path = directory / FILES["finalize"]
    if path.exists():
        rows = load_rows(path)
        summary["final_sufficient"] = sum(r["evidence_state"]["status"] == "sufficient" for r in rows)
        summary["best_available_fallback"] = sum(r["best_available_fallback"] for r in rows)
        summary["max_external_followup_rounds"] = max((r["external_followup_rounds"] for r in rows), default=0)
        summary["average_selected"] = sum(len(r["selected_evidence"]) for r in rows) / max(1, len(rows))
        summary["uncertain_selected"] = sum(e["esa_label"] == "uncertain" for r in rows for e in r["selected_evidence"])
        summary["final_failure_reasons"] = {reason: sum(r["stop_reason"] == reason for r in rows)
                                            for reason in sorted({r["stop_reason"] for r in rows})}
    path = directory / FILES["plan"]
    if path.exists():
        rows = load_rows(path)
        summary["plan_fallback_records"] = sum(r["plan_audit"]["status"] == "fallback" for r in rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary

def preflight_assets(args):
    for flag in ("qwen_model", "ce_model", "query_model"):
        path = Path(getattr(args, flag))
        if not (path / "config.json").is_file():
            raise FileNotFoundError("Missing local model config for --%s: %s" % (flag, path))
    if not (Path(args.medrag_repo) / "src/utils.py").is_file():
        raise FileNotFoundError("--medrag_repo does not contain src/utils.py")
    for corpus in CORPORA[args.corpus]:
        root = Path(args.db_dir) / corpus
        if not any((root / "chunk").glob("*.jsonl")):
            raise FileNotFoundError("Missing chunk JSONL: " + str(root))
        for name in ("faiss.index", "metadatas.jsonl"):
            if not (root / "index/ncbi/MedCPT-Article-Encoder" / name).is_file():
                raise FileNotFoundError("Missing MedCPT index for %s; run --stage prepare_indexes before planning" % corpus)

def main():
    args = parser().parse_args()
    directory = Path(args.run_dir).resolve()
    if args.stage == "audit":
        audit_run(directory)
        return
    if not 1 <= args.min_selected <= args.max_selected <= 12:
        raise ValueError("Require 1 <= min_selected <= max_selected <= 12")
    for key in ("pool_size", "per_query_k", "pool_batch", "pool_ce_topk_per_gap",
                "context_chars", "evidence_chars", "rrf_k"):
        if getattr(args, key) <= 0:
            raise ValueError(key + " must be positive")
    if args.selection_stop_gain < 0 or not 0 <= args.near_duplicate_threshold <= 1:
        raise ValueError("invalid evidence selection thresholds")
    if args.evidence_chars > args.context_chars:
        raise ValueError("evidence_chars must not exceed context_chars")
    if args.stage == "prepare_indexes":
        from src.evidence_gap_models import MedTextRetriever
        MedTextRetriever(args.medrag_repo, args.db_dir, args.corpus, args.query_model,
                         args.article_model, prepare=True)
        print("MedCPT indexes ready for", args.corpus)
        return
    if args.stage == "all":
        preflight_assets(args)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = get_manifest(args)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Run config/input/code changed. Use a NEW --run_dir, do not mix experimental artifacts.")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.stage == "all":
        argv = sys.argv[1:]
        clean, skip = [], False
        for token in argv:
            if skip:
                skip = False
                continue
            if token == "--stage":
                skip = True
            elif not token.startswith("--stage="):
                clean.append(token)
        for stage in STAGES:
            python = args.retrieval_python if stage in {"retrieve", "followup"} and args.retrieval_python else sys.executable
            subprocess.run([python, str(Path(__file__).resolve()), *clean, "--stage", stage], check=True)
        audit_run(directory)
        return

    def read(stage):
        return load_rows(directory / FILES[stage])

    models = {}

    def qwen():
        if "qwen" not in models:
            from src.evidence_gap_models import QwenJSON
            models["qwen"] = QwenJSON(args.qwen_model, args.device, args.qwen_context)
        return models["qwen"]

    def retriever():
        if "retriever" not in models:
            from src.evidence_gap_models import MedTextRetriever
            models["retriever"] = MedTextRetriever(args.medrag_repo, args.db_dir, args.corpus,
                                                    args.query_model, args.article_model)
        return models["retriever"]

    def ce():
        if "ce" not in models:
            from src.evidence_gap_models import CrossEncoder
            models["ce"] = CrossEncoder(args.ce_model, args.device, args.ce_batch_size)
        return models["ce"]

    if args.stage == "plan":
        from tools.export_medrag_snippets import load_kgarevion_dataset
        examples = load_kgarevion_dataset(args.dataset, args.benchmark_path)
        requested_source_keys = load_source_keys(args.source_keys_path)
        if requested_source_keys is not None:
            available = {str(example["source_key"]) for example in examples}
            missing = sorted(requested_source_keys - available)
            if missing:
                raise ValueError("Unknown source keys requested: %s" % missing[:10])
            examples = [example for example in examples
                        if str(example["source_key"]) in requested_source_keys]
        if args.eval_start_index < 0 or args.max_samples == 0 or args.max_samples < -1:
            raise ValueError("Invalid sample range")
        examples = examples[args.eval_start_index:]
        if args.max_samples > 0:
            examples = examples[:args.max_samples]
        if not examples:
            raise ValueError("Empty benchmark slice")
        inputs = [dict(index=e["index"], source_key=e["source_key"], dataset=args.dataset,
                       question=e["question"], options=e["options"]) for e in examples]

        def operation(row):
            result, audit = qwen().ask(plan_prompt(row, args.num_views),
                                       lambda d: validate_plan(d, args.num_views), args.plan_tokens)
            plan = result or {"observed_clues": [], "asked_aspect": "Answer the question",
                              "information_needs": [], "query_views": []}
            return dict(row, pipeline=PIPELINE, plan=plan, plan_audit=audit,
                        retrieval_queries=initial_queries(row["question"], plan),
                        retrieval_contract={"corpus": args.corpus, "components": CORPORA[args.corpus],
                                            "retriever": "MedCPT", "db_dir": str(Path(args.db_dir).resolve())})
    elif args.stage in {"retrieve", "followup"}:
        inputs = read("plan" if args.stage == "retrieve" else "analyze")

        def operation(row):
            queries = row["retrieval_queries"] if args.stage == "retrieve" else row["bridge_queries"]
            pool = retriever().retrieve(queries, args.per_query_k, args.rrf_k) if queries else []
            return dict(base_row(row), snippets=pool, retrieval_query_count=len(queries))
    elif args.stage == "rerank":
        inputs = read("retrieve")

        def operation(row):
            candidates = ce().rerank(row, row["snippets"])[:args.pool_size] if row["snippets"] else []
            return dict(base_row(row), snippets=candidates)
    elif args.stage == "select":
        inputs = read("rerank")

        def operation(row):
            return dict(base_row(row), selected_evidence=selection(row, row["snippets"], args),
                        selection_config={"max_selected": args.max_selected, "min_selected": args.min_selected,
                                          "context_chars": args.context_chars, "evidence_chars": args.evidence_chars,
                                          "stop_gain": args.selection_stop_gain,
                                          "near_duplicate_threshold": args.near_duplicate_threshold},
                        selection_phase="initial_no_bridge", prompt_alignment_pairs=[], conflict_notes=[])
    elif args.stage == "analyze":
        inputs = joined(read("rerank"), read("select"))

        def operation(pair):
            pool_ce = ce() if args.pool_check_backend == "cross_encoder" else None
            return analyze_record(pair["left"], pair["right"], qwen(), args, pool_ce)
    elif args.stage == "merge_rerank":
        inputs = joined(read("rerank"), read("followup"))

        def operation(pair):
            first, second = pair["left"], pair["right"]
            old = first["snippets"]
            old_ids = {e["evidence_id"] for e in old}
            new = second["snippets"]
            # Score new-only content; keep identical evidence's original frozen score.
            novel = [e for e in new if e["evidence_id"] not in old_ids]
            scored = ce().rerank(second, novel) if novel else []
            pool = merge_candidates(old, new, scored)
            for query in second["bridge_queries"]:
                for e, score in zip(pool, ce().score(query["query"], pool)):
                    e.setdefault("bridge_query_logits", {})[query["name"]] = score
            return dict(base_row(second), snippets=pool, new_unique_candidates=len(novel))
    else:
        inputs = joined(read("analyze"), read("merge_rerank"))

        def operation(pair):
            a, b = pair["left"], pair["right"]
            return finalize_record(a, b, qwen() if a["route"] == "retrieve" else None, args)
    run_cached(directory / FILES[args.stage], inputs, operation)

if __name__ == "__main__":
    main()
