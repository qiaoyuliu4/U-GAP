"""U-GAP runtime components extracted from the research implementation."""

import copy

import hashlib

import json

import math

import re

PIPELINE = "evidence_gap_v1"

CORPORA = {"MedText": ["textbooks", "statpearls"],
           "Textbooks": ["textbooks"], "StatPearls": ["statpearls"]}

LABELS = {"useful", "uncertain", "irrelevant"}

def norm(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()

def tokens(value):
    return set(re.findall(r"[a-z0-9]+", norm(value).lower()))

def jaccard(left, right):
    a, b = tokens(left), tokens(right)
    return len(a & b) / max(1, len(a | b))

def split_sentences(value):
    """Return stable sentence IDs without requiring brittle verbatim generation."""
    text = norm(value)
    parts = [norm(x) for x in re.split(r"(?<=[.!?])\s+|\s*[\r\n]+\s*", text) if norm(x)]
    return [{"sentence_id": "S%d" % (i + 1), "text": part} for i, part in enumerate(parts)]

def redundancy_score(left, right):
    """Catch both whole-chunk overlap and repeated sentences in adjacent chunks."""
    score = jaccard(left, right)
    left_sentences = [x["text"] for x in split_sentences(left) if len(tokens(x["text"])) >= 5]
    right_sentences = [x["text"] for x in split_sentences(right) if len(tokens(x["text"])) >= 5]
    for a in left_sentences:
        for b in right_sentences:
            score = max(score, jaccard(a, b))
    return score

def identity(row):
    return str(row.get("dataset", "")), int(row["index"])

def assert_same_question(left, right):
    if identity(left) != identity(right) or norm(left["question"]) != norm(right["question"]):
        raise ValueError("Question identity mismatch: %s / %s" % (identity(left), identity(right)))
    if left.get("options", {}) != right.get("options", {}):
        raise ValueError("Choices mismatch for %s" % (identity(left),))

def content(item):
    for key in ("content", "text", "contents", "passage", "abstract"):
        if isinstance(item.get(key), str) and item[key].strip():
            return norm(item[key])
    return ""

def excerpt(text, limit):
    text = norm(text)
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0]

def evidence_id(item):
    # Full content, not its first 500 characters; shared text keeps all provenance.
    return "T" + hashlib.sha256(content(item).lower().encode("utf-8")).hexdigest()[:16]

def merge_candidates(*pools):
    merged = {}
    for pool in pools:
        for raw in pool:
            if not content(raw):
                continue
            item = copy.deepcopy(raw)
            eid = evidence_id(item)
            item.update(evidence_id=eid, content=content(item), evidence_type="text")
            if eid not in merged:
                merged[eid] = item
                continue
            old = merged[eid]
            for key in ("retrieval_sources", "provenance_records"):
                values = old.setdefault(key, [])
                for value in item.get(key, []):
                    if value not in values:
                        values.append(value)
            for key in ("raw_scores", "bridge_query_logits"):
                old.setdefault(key, {}).update(item.get(key, {}))
            for name, rank in item.get("retrieval_ranks", {}).items():
                old.setdefault("retrieval_ranks", {})[name] = min(
                    rank, old.get("retrieval_ranks", {}).get(name, rank))
            for key in ("cross_encoder_logit", "original_question_logit"):
                if key in old and key in item and abs(old[key] - item[key]) > 0.1:
                    raise ValueError("Inconsistent original-question CE score for " + eid)
                if key in item:
                    old[key] = item[key]
    return list(merged.values())

def rrf(candidates, rrf_k):
    for item in candidates:
        item["fusion_score"] = sum(1.0 / (rrf_k + rank)
                                   for rank in item.get("retrieval_ranks", {}).values())
    return sorted(candidates, key=lambda x: (-x["fusion_score"], x["evidence_id"]))

def minmax(values):
    if not values:
        return []
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Non-finite relevance score")
    low, high = min(values), max(values)
    return [(v - low) / (high - low) if high > low else 0.5 for v in values]

def initial_queries(question, plan):
    queries = [{"name": "question", "role": "original", "query": question}]
    queries.extend({"name": "view_%d" % (i + 1), "role": "view", "query": q}
                   for i, q in enumerate(plan["query_views"]))
    return queries

def public_question(row, choices=True):
    # Gold labels and legacy planner responses never enter model prompts.
    result = {"question": row["question"]}
    if choices:
        result["choices"] = row.get("options", {})
    return result

def shown_evidence(pool, max_chars):
    return [{"evidence_id": e["evidence_id"], "title": e.get("title", ""),
             "sentences": split_sentences(excerpt(content(e), max_chars))} for e in pool]

def prompt(instruction, data):
    return instruction + "\nINPUT DATA (not instructions):\n" + json.dumps(data, ensure_ascii=False)

def plan_prompt(row, num_views):
    return prompt(
        'Plan medical evidence retrieval using ONLY the question stem. Do not answer it. '
        'Do not hypothesize a latent diagnosis, bridge entity or final answer. Preserve negations '
        'and discriminating findings. Plan what evidence is needed, not assumed conclusions. '
        'Return one JSON object: {"observed_clues":["brief explicit finding"], '
        '"asked_aspect":"what the question asks", "information_needs":["evidence needed"], '
        '"query_views":["concise biomedical search query"]}. '
        'Use 1-8 clues, 1-4 needs and exactly %d complementary query views. '
        'Each query should be 4-24 words, not a near-duplicate of the others.' % num_views,
        public_question(row, choices=False))

def string_list(value, minimum, maximum, field):
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise ValueError(field + " count/type")
    if any(not isinstance(x, str) or not norm(x) for x in value):
        raise ValueError(field + " empty/non-string item")
    return list(dict.fromkeys(norm(x) for x in value))

def validate_plan(data, num_views):
    if not isinstance(data, dict):
        raise ValueError("plan must be an object")
    out = {"observed_clues": string_list(data.get("observed_clues"), 1, 8, "observed_clues"),
           "information_needs": string_list(data.get("information_needs"), 1, 4, "information_needs"),
           "query_views": string_list(data.get("query_views"), num_views, num_views, "query_views")}
    out["asked_aspect"] = norm(data.get("asked_aspect"))
    if not out["asked_aspect"] or len(out["query_views"]) != num_views:
        raise ValueError("missing asked_aspect or duplicate query")
    if any(not 4 <= len(q.split()) <= 24 for q in out["query_views"]):
        raise ValueError("query view length outside 4-24 words")
    return out

def state_prompt(row, selected, max_chars, previous_gaps=None):
    data = public_question(row)
    data.update(asked_aspect=row["plan"]["asked_aspect"],
                information_needs=row["plan"]["information_needs"],
                evidence=shown_evidence(selected, max_chars),
                previous_gaps=previous_gaps or [])
    return prompt(
        'Analyze decision sufficiency: whether the evidence is adequate to support a defensible '
        'answer to the asked aspect and, when choices exist, distinguish the clinically plausible '
        'choices. Choices are hypotheses, NOT facts. Direct evidence supporting one decision can be '
        'sufficient; do NOT demand a separate review of every choice, exhaustive treatment details, '
        'or evidence for incidental distractor clues. For yes/no/maybe, judge the specific '
        'claim and study context; absence of evidence is not evidence for no or maybe. '
        'For treatment-priority or next-step questions, symptom matching or identifying the '
        'syndrome alone is NOT sufficient: at least one citation must directly support the '
        'requested priority, initial action, or clinically discriminating relation among the '
        'plausible choices. Evidence about a complication outside the listed choices does not '
        'establish decision sufficiency. '
        'Do not output an answer or solve using unsupported medical assumptions. '
        'Judge each evidence item useful / uncertain / irrelevant. Uncertain is not irrelevant. '
        'Previous gaps are hypotheses, not mandatory requirements. Reassess each as resolved, '
        'unresolved, or not_needed; do not preserve a gap merely because it was previously proposed. '
        'If previous_gaps is empty, gap_assessments MUST be an empty list; put newly identified '
        'gaps only in missing_relations. '
        'Identify up to TWO decision-critical missing relations, not generic requests for more '
        'information. A new gap is decision-critical only if retrieving it could change which '
        'choice is preferred or directly establish the asked aspect. Do not create a separate gap '
        'for an incidental symptom or behavior unless it selects among or contraindicates the '
        'plausible choices. Cite supplied sentence IDs for supported needs. '
        'Return JSON: {"status":"sufficient|insufficient|uncertain", '
        '"evidence_judgments":[{"evidence_id":"T...","label":"useful|uncertain|irrelevant",'
        '"reason":"brief"}], "supported_needs":[{"need":"...","evidence_id":"T...",'
        '"sentence_ids":["S1"]}], "gap_assessments":[{"gap_id":"gap1",'
        '"status":"resolved|unresolved|not_needed","evidence_ids":["T..."],"reason":"brief"}], '
        '"missing_relations":[{"id":"gap1",'
        '"relation":"entity/relation or specific evidence missing", "why_needed":"brief"}], '
        '"failure_reason":"empty if sufficient; otherwise brief"}. '
        'Sufficient requires cited evidence and no unresolved missing relation. '
        'If unsure about sufficiency, say uncertain, without inventing a gap.', data)

def valid_citations(items, pool, max_chars):
    texts = {e["evidence_id"]: norm(excerpt(content(e), max_chars)).lower() for e in pool}
    sentence_maps = {e["evidence_id"]: {x["sentence_id"]: x["text"] for x in
                     split_sentences(excerpt(content(e), max_chars))} for e in pool}
    valid = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        eid = item.get("evidence_id")
        if eid not in texts:
            continue
        sentence_ids = item.get("sentence_ids", [])
        if isinstance(sentence_ids, list) and sentence_ids and all(
                sid in sentence_maps[eid] for sid in sentence_ids):
            clean = copy.deepcopy(item)
            clean["sentence_ids"] = list(dict.fromkeys(sentence_ids))[:3]
            clean["quote"] = " ".join(sentence_maps[eid][sid] for sid in clean["sentence_ids"])
            valid.append(clean)
            continue
        # Read old audit fixtures, but new prompts use sentence IDs.
        quote = norm(item.get("quote", "")).lower()
        if len(quote) >= 12 and quote in texts[eid]:
            valid.append(copy.deepcopy(item))
    return valid

def validate_state(data, selected, max_chars, previous_gaps=None):
    if not isinstance(data, dict) or data.get("status") not in {"sufficient", "insufficient", "uncertain"}:
        raise ValueError("invalid ESA status")
    by_id = {e["evidence_id"] for e in selected}
    judgments = {}
    items = data.get("evidence_judgments", [])
    if not isinstance(items, list):
        raise ValueError("evidence_judgments must be a list")
    for item in items:
        if not isinstance(item, dict) or item.get("evidence_id") not in by_id or item.get("label") not in LABELS:
            raise ValueError("invalid evidence ID or soft label")
        if item["evidence_id"] in judgments:
            raise ValueError("duplicate evidence judgment")
        judgments[item["evidence_id"]] = item
    # A single omitted item must not collapse the whole route. Preserve it as
    # uncertain so the soft-label policy remains conservative and auditable.
    for evidence in selected:
        eid = evidence["evidence_id"]
        if eid not in judgments:
            judgments[eid] = {"evidence_id": eid, "label": "uncertain",
                              "reason": "not_explicitly_judged"}
    previous_gaps = previous_gaps or []
    previous_by_id = {g["id"]: g for g in previous_gaps}
    assessments = data.get("gap_assessments", [])
    if not isinstance(assessments, list):
        raise ValueError("gap_assessments must be a list")
    if previous_by_id and {x.get("gap_id") for x in assessments if isinstance(x, dict)} != set(previous_by_id):
        raise ValueError("ESA must reassess every previous gap")
    # Qwen sometimes repeats newly proposed gaps in gap_assessments during the
    # first analysis. They are redundant with missing_relations, not a reason to
    # discard an otherwise valid medical assessment.
    if not previous_by_id:
        assessments = []
    cleaned_assessments, unresolved = [], []
    for item in assessments:
        if not isinstance(item, dict) or item.get("status") not in {"resolved", "unresolved", "not_needed"}:
            raise ValueError("invalid previous-gap assessment")
        evidence_ids = item.get("evidence_ids", [])
        if not isinstance(evidence_ids, list) or any(eid not in by_id for eid in evidence_ids):
            raise ValueError("invalid gap-assessment evidence ID")
        if item["status"] == "resolved" and not evidence_ids:
            raise ValueError("resolved gap requires evidence")
        cleaned_assessments.append({"gap_id": item["gap_id"], "status": item["status"],
                                    "evidence_ids": list(dict.fromkeys(evidence_ids)),
                                    "reason": norm(item.get("reason"))})
        if item["status"] == "unresolved":
            unresolved.append(copy.deepcopy(previous_by_id[item["gap_id"]]))

    gaps = data.get("missing_relations", [])
    if not isinstance(gaps, list) or len(gaps) > 2:
        raise ValueError("too many gaps")
    cleaned = unresolved
    previous_relations = {norm(g["relation"]).lower() for g in previous_gaps}
    seen_relations = set(previous_relations)
    for gap in gaps:
        if not isinstance(gap, dict) or not norm(gap.get("relation")):
            raise ValueError("missing relation text")
        relation = norm(gap["relation"])
        if relation.lower() in seen_relations:
            continue
        if len(cleaned) < 2:
            cleaned.append({"id": "gap%d" % (len(cleaned) + 1), "relation": relation,
                            "why_needed": norm(gap.get("why_needed"))})
            seen_relations.add(relation.lower())
    for i, gap in enumerate(cleaned):
        gap["id"] = "gap%d" % (i + 1)
    citations = valid_citations(data.get("supported_needs", []), selected, max_chars)
    if data["status"] == "sufficient" and (cleaned or not citations):
        raise ValueError("unsupported sufficient judgment")
    failure_reason = norm(data.get("failure_reason"))
    if data["status"] == "insufficient" and (not cleaned or not failure_reason):
        raise ValueError("insufficient requires an actionable gap and failure reason")
    return {"status": data["status"], "evidence_judgments": [
                judgments[e["evidence_id"]] for e in selected],
            "supported_needs": citations, "gap_assessments": cleaned_assessments,
            "missing_relations": cleaned, "failure_reason": failure_reason}

def unknown_state(selected, reason):
    return {"status": "uncertain", "evidence_judgments": [
        {"evidence_id": e["evidence_id"], "label": "uncertain", "reason": reason} for e in selected],
        "supported_needs": [], "gap_assessments": [], "missing_relations": [], "failure_reason": reason}

def pool_prompt(row, gaps, batch, max_chars):
    data = public_question(row)
    data.update(missing_relations=gaps, evidence=shown_evidence(batch, max_chars))
    return prompt(
        'Check existing candidate evidence for EACH missing relation. Mere shared words, a disease '
        'name, or a choice mentioned in text does NOT establish the needed relation. '
        'Use covered only if the provided evidence actually supplies it; otherwise uncertain or absent. '
        'Return JSON {"gap_checks":[{"gap_id":"gap1","status":"covered|uncertain|absent",'
        '"matches":[{"evidence_id":"T...","sentence_ids":["S1"], "reason":"brief"}]}]}. '
        'Check every supplied gap. Cite supplied sentence IDs; no final answer.', data)

def validate_pool_check(data, gaps, batch, max_chars):
    valid_ids = {g["id"] for g in gaps}
    checks = data.get("gap_checks", []) if isinstance(data, dict) else []
    if not isinstance(checks, list) or len(checks) != len(valid_ids):
        raise ValueError("gap check count/type")
    if {x.get("gap_id") for x in checks if isinstance(x, dict)} != valid_ids:
        raise ValueError("missing/unknown gap checks")
    result = []
    for check in checks:
        if check.get("status") not in {"covered", "uncertain", "absent"}:
            raise ValueError("invalid gap status")
        matches = valid_citations(check.get("matches", []), batch, max_chars)
        status = check["status"]
        if status == "covered" and not matches:
            raise ValueError("covered gap without valid quotation")
        result.append({"gap_id": check["gap_id"], "status": status, "matches": matches})
    return result

def bridge_prompt(row, gaps):
    data = public_question(row)
    data.update(missing_relations=gaps, asked_aspect=row["plan"]["asked_aspect"])
    return prompt(
        'Generate at most TWO targeted biomedical search queries to find the supplied missing '
        'relations. Preserve the original question constraints. Do not assume any choice is correct. '
        'Use concrete clinical entities and the relation sought. Do not merely restate the question, '
        'write generic phrases such as "management" or "specific evidence", or list more than two '
        'choices in one query. For intervention comparisons, prefer one focused comparison or one '
        'query per leading intervention pair. Write a compact search query, not a sentence beginning '
        'with "What is" or "What is the evidence". Example form: "febrile flank pain dysuria '
        'initial evaluation urinalysis urine culture CT indications". No final answer and no new '
        'diagnostic assertions. '
        'Return JSON {"bridge_queries":[{"gap_id":"gap1","query":"4-24 words",'
        '"rationale":"what missing evidence to retrieve"}]}. Empty list is allowed if no '
        'safe actionable search can be formulated.', data)

def validate_bridges(data, gaps, options=None):
    items = data.get("bridge_queries") if isinstance(data, dict) else None
    if not isinstance(items, list) or len(items) > 2:
        raise ValueError("bridge query count/type")
    ids = {g["id"] for g in gaps}
    out, seen = [], set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("bridge query must be an object")
        query = norm(item.get("query"))
        if item.get("gap_id") not in ids:
            raise ValueError("bridge query references an unknown gap_id")
        words = query.split()
        if not 4 <= len(words) <= 24:
            raise ValueError(
                "bridge query has %d words; rewrite it as 4-24 keyword-rich words without a full "
                "patient restatement" % len(words))
        lowered = query.lower().lstrip(" ?")
        if lowered.startswith(("what is ", "what are ", "what is the evidence", "specific evidence")):
            raise ValueError(
                "bridge query is a generic question; rewrite it as concrete entities plus the "
                "missing relation, without 'what is' or 'specific evidence'")
        option_mentions = sum(norm(value).lower() in lowered for value in (options or {}).values()
                              if norm(value))
        if option_mentions > 2:
            raise ValueError(
                "bridge query lists more than two choices; use one focused comparison or split it")
        if query.lower() not in seen:
            out.append({"name": "bridge_%d" % (len(out) + 1), "role": "bridge",
                        "gap_id": item["gap_id"], "query": query,
                        "rationale": norm(item.get("rationale"))})
            seen.add(query.lower())
    return out

def select_evidence(row, pool, max_selected=8, min_selected=4, max_chars=8000,
                    per_evidence_chars=1200, gaps=None, gap_support=None, judgments=None,
                    stop_gain=0.60, near_duplicate_threshold=0.85):
    """Greedy marginal coverage gain plus CE relevance minus lexical redundancy.

    First pass has no gap term. Uncertain evidence has exactly the same prior as
    unjudged evidence; soft labels never act as an eligibility mask.
    """
    candidates = merge_candidates(pool)
    relevance = minmax([float(e["original_question_logit"]) for e in candidates])
    views = {q["name"] for q in row["retrieval_queries"] if q["role"] == "view"}
    gap_ids = {g["id"] for g in gaps or []}
    gap_support = gap_support or {}
    judgments = judgments or {}
    gap_values = {e["evidence_id"]: dict(gap_support.get(e["evidence_id"], {})) for e in candidates}
    # Dual CE relevance is only a retrieval proxy, never medical entailment.
    # Bridge relevance must be allowed to recover narrow relation evidence that
    # is not highly similar to the full original vignette.
    for query in row.get("bridge_queries", []):
        available = [e for e in candidates if query["name"] in e.get("bridge_query_logits", {})]
        scores = minmax([e["bridge_query_logits"][query["name"]] for e in available])
        for e, score in zip(available, scores):
            gid = query["gap_id"]
            if gid in gap_ids:
                gap_values[e["evidence_id"]][gid] = max(
                    gap_values[e["evidence_id"]].get(gid, 0), score)
    selected, covered_views, covered_gaps, used_chars = [], set(), {}, 0
    remaining = list(zip(candidates, relevance))
    while remaining and len(selected) < max_selected:
        choices = []
        for e, rel in remaining:
            text = excerpt(content(e), per_evidence_chars)
            if not text or used_chars + len(text) > max_chars:
                continue
            ev_views = views & set(e.get("retrieval_sources", []))
            view_gain = len(ev_views - covered_views) / max(1, len(views))
            # Covering one decision-critical gap should retain its full value;
            # averaging by the number of gaps made targeted evidence vanish as
            # soon as ESA proposed two gaps. Cap the total so multi-gap matches
            # cannot inflate a candidate without bound.
            gap_gain = min(1.0, sum(max(0, v - covered_gaps.get(g, 0))
                                   for g, v in gap_values[e["evidence_id"]].items()
                                   if g in gap_ids))
            redundancy = max((redundancy_score(text, s["content"]) for s in selected), default=0.0)
            label = judgments.get(e["evidence_id"], "uncertain")
            soft = {"useful": 0.05, "uncertain": 0.0, "irrelevant": -0.25}[label]
            gain = rel + 0.45 * view_gain - 0.30 * redundancy + soft
            if gap_ids:
                gain += 1.00 * gap_gain
            if redundancy >= near_duplicate_threshold:
                gain -= 0.45
            choices.append((gain, e["evidence_id"], e, rel, text, ev_views))
        if not choices:
            break
        gain, eid, e, rel, text, ev_views = max(choices, key=lambda x: (x[0], x[1]))
        if gain < stop_gain and len(selected) >= min_selected:
            break
        chosen = copy.deepcopy(e)
        chosen.update(content=text, relevance_score=rel, selection_score=gain,
                      esa_label=judgments.get(eid, "uncertain"),
                      provenance={"title": e.get("title", ""), "records": e.get("provenance_records", [])})
        selected.append(chosen)
        used_chars += len(text)
        covered_views.update(ev_views)
        for g, value in gap_values[eid].items():
            covered_gaps[g] = max(covered_gaps.get(g, 0), min(rel, value))
        remaining = [(x, r) for x, r in remaining if x["evidence_id"] != eid]
    return selected

def apply_final_state(row, selected, state, stop_reason, retrieval_rounds):
    out = copy.deepcopy(row)
    by_id = {j["evidence_id"]: j["label"] for j in state["evidence_judgments"]}
    annotated = [dict(e, esa_label=by_id.get(e["evidence_id"], "uncertain")) for e in selected]
    # Preserve uncertain items. If all are irrelevant, explicitly keep the best
    # available set as a failed-evidence fallback rather than silently answer empty.
    retained = [e for e in annotated if e["esa_label"] != "irrelevant"]
    fallback = not retained
    out.update(pipeline=PIPELINE, selected_evidence=retained or annotated,
               evidence_state=state, stop_reason=stop_reason,
               external_followup_rounds=retrieval_rounds,
               best_available_fallback=fallback or state["status"] != "sufficient",
               failure_reason="" if state["status"] == "sufficient" and not fallback else
               (state.get("failure_reason") or stop_reason),
               prompt_alignment_pairs=[], conflict_notes=[])
    if retrieval_rounds not in (0, 1):
        raise ValueError("At most one external follow-up is allowed")
    return out
