"""Evaluate saved U-GAP predictions without loading a model."""
import argparse
from collections import Counter
import json
from pathlib import Path

from src.uncertainty_gate import candidate_space, canonical_answer


def evaluate(path, expected_total=None):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not rows or (expected_total is not None and len(rows) != expected_total):
        raise ValueError("Empty/incomplete predictions: got %d, expected %s" % (len(rows), expected_total))
    identities, indices, correct, scored, invalid = set(), set(), 0, 0, 0
    routes = Counter()
    for row in rows:
        key = (row["dataset"], str(row["source_key"]))
        index = (row["dataset"], int(row["index"]))
        if key in identities or index in indices:
            raise ValueError("Duplicate source key or index: " + str(key))
        identities.add(key)
        indices.add(index)
        space = candidate_space(row, row["dataset"])
        prediction = row.get("final_prediction")
        if prediction not in space["labels"]:
            invalid += 1
        gold = canonical_answer(row, space)
        if gold:
            if gold not in space["labels"]:
                raise ValueError("Invalid gold answer: " + str(key))
            scored += 1
            correct += prediction == gold
        routes[row.get("final_route", "unknown")] += 1
    return {"records": len(rows), "labeled_records": scored, "correct": correct,
            "accuracy": correct / scored if scored else None, "invalid_predictions": invalid,
            "routes": dict(routes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--expected-total", type=int)
    parser.add_argument("--output", help="New metrics JSON file; existing files are never replaced")
    args = parser.parse_args()
    metrics = evaluate(args.predictions, args.expected_total)
    text = json.dumps(metrics, indent=2)
    print(text)
    if metrics["accuracy"] is not None:
        print("Accuracy: %d/%d = %.3f%%" % (metrics["correct"], metrics["labeled_records"], 100 * metrics["accuracy"]))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
