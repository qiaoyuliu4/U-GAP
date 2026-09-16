# Attribution and redistribution status

U-GAP was developed by modifying the KGARevion research codebase. It is not an
official KGARevion release, and the authorship of inherited code is not claimed
by the U-GAP maintainers. Core model-loading/answer interfaces and the benchmark
format derive from that work. `docs/source_map.json` maps retained modules to
their research implementation locations.

## KGARevion

- Repository: https://github.com/mims-harvard/KGARevion
- Xiaorui Su, Yibo Wang, Shanghua Gao, Xiaolong Liu, Valentina Giunchiglia,
  Djork-Arne Clevert, and Marinka Zitnik. *KGARevion: An AI Agent for
  Knowledge-Intensive Biomedical QA*. ICLR 2025.
- Upstream license not located during preparation on 2026-09-15. Obtain written
  permission or the applicable upstream license before redistribution; preserve
  the received notice verbatim. The source tree here is a local release candidate.

## MedRAG and MedCPT

- MedRAG: https://github.com/gzxiong/MedRAG
- MedCPT: https://github.com/ncbi/MedCPT
- MedRAG is installed separately. Its source/license, model checkpoints, corpus
  and indexes are not copied into this repository. Retain their upstream notices
  when preparing an environment or redistributing permitted assets.

## Models and data

- Llama 3: obtain weights from the authorized Meta distribution and accept its terms.
- Qwen3: obtain the checkpoint and its accompanying license from its publisher.
- Benchmarks and MedText components: obtain them from their respective providers;
  downloading a dataset does not imply permission to redistribute its contents.
- `examples/sample_questions.jsonl` contains newly written illustrative questions,
  not copied benchmark records. They are for software checks, not benchmark claims.

Python dependencies retain their own licenses. No API credentials, pretrained
weights, benchmark answers, corpus text, or historical experimental logs are bundled.
