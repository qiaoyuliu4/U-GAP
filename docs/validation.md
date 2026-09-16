# Validation and Release Checklist

## Completed Locally

The standard-library test suite runs without model weights:

```bash
python -m unittest discover -s tests -v
```

It checks:

1. The extracted package can be relocated outside the research tree and validate its inputs independently.
2. Duplicate IDs, illegal semantic option mappings and document budgets are checked.
3. A new temporary output directory runs Gate, planning, retrieval, ranking, evidence construction, both ESA passes, G5, fusion and final evaluation using **synthetic model/retriever doubles**.
4. Repeating that run resumes its own completed records without new model calls.

The synthetic test verifies orchestration and data contracts, not real medical quality, numerical model equivalence, dependency installation or GPU memory use. No historical experiment predictions are used in the test. Test outputs live in temporary directories and are not included in the release.

## Still Required on a GPU Server

The README fresh-small-run commands must be executed using actual checkpoints, a compatible external MedRAG checkout, and real prepared indexes. This was **not executed locally** because the local packaging environment has no PyTorch/Transformers installation or model weights.

Before marking this release runnable-and-verified:

- Install both requirements files in new environments; record the actual resolved packages.
- Set all asset paths in a private local config and run the no-write `--check`.
- Run all stages on the three example questions in a never-used output directory, with threshold 0.0 for the forced-RAG wiring check.
- Evaluate `07_final_predictions.jsonl`; require three unique labeled records and zero invalid final predictions. Do not require an inflated toy accuracy target.
- Inspect final routes, intermediate artifacts and logs. It is valid for real examples not to trigger G5/fusion; the synthetic suite covers those paths, while a routed real input is needed to validate their actual GPU execution.
- Restart the same command and verify no unnecessary inference is repeated.
- Freeze a source snapshot and the validated dependency/asset manifests before benchmark reporting.

## Public Distribution Gate

- Confirm redistribution permission for KGARevion-derived code and preserve required notices. An absent upstream license is not permission to label inherited code MIT.
- Choose an appropriate license for original additions only after resolving inherited-code obligations; the current LICENSE records the unresolved status.
- Remove local configs, environment exports containing private paths, weights, corpus, indexes, predictions and logs from the public staging area.
- Preserve `THIRD_PARTY_NOTICES.md`, prompt implementations and fallback behavior.
- State test-set tuning/exploratory-evaluation limitations; do not present historical tuned test results as an untouched test evaluation.

The current directory is therefore a **locally tested release candidate**, not a claim that redistribution authorization or the real-GPU release acceptance test is complete.
