# Packaging Changes Relative to the Research Tree

This directory is a separate release candidate. The original research project and experiment outputs were not cleaned in place or deleted.

## Preserved

- Gate scores, normalized entropy, label mapping and comparison rule.
- Planning/ESA/Bridge prompts, validators and failed-analysis fallbacks.
- Retrieval/ranking/selection logic, final-state filtering and ID checks.
- Direct-derived Reader task instructions, option formatting and answer parser.
- G5 prompt construction, no-options generator input, sampling defaults and per-document seeds.
- Conflict-aware source separation, agreement routing and invalid-output fallback order.

## Release-Specific Changes

- Extracted required runtime definitions instead of shipping all experimental runners. Existing names were retained where useful; see `source_map.json`.
- Removed unused Azure/OpenAI branches and local Llama checkpoint fallback paths from the extracted model wrapper.
- Added `run.py` configuration/path normalization, portable question input, count/ID checks, and explicit index preparation dispatch.
- Made entropy threshold and generated-document count configurable, retaining defaults 0.02 and 5. Counts smaller than retained gap needs fail explicitly.
- Expanded manifest source coverage to all extracted runtime Python modules plus the public entry point.
- Added completeness checks before fusion/finalization so a skipped G5 stage cannot be mistaken for a legitimate missing-gap fallback.
- Added an independent evaluation CLI, original toy inputs, installation/data documentation and offline contract tests.

This extraction is not asserted to be byte-identical to a historical execution snapshot. Default scientific behavior is intended to be preserved, but a matched real-model small-run comparison is still required before claiming numerical equivalence. Reuse of old research output directories is intentionally rejected.

## Data and Privacy Boundary

The package contains no benchmark test set, prior predictions, old manifests, old debug logs, corpus/index assets, model weights, API keys or user-machine path bindings. New runs naturally save user-specified local paths in their private manifests; review those generated files before sharing them publicly.
