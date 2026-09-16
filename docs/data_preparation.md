# Data, Models and Indexes

Prepare assets before running U-GAP. Inference uses local checkpoints; normal runs do not download a corpus or silently rebuild indexes. Do not commit assets to this repository.

## 1. Checkpoints

Obtain complete checkpoints through the official project/model repositories, after accepting applicable access conditions:

| Configuration key | Resource |
| --- | --- |
| `llama_model` | [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |
| `qwen_model` | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| `query_model` | [ncbi/MedCPT-Query-Encoder](https://huggingface.co/ncbi/MedCPT-Query-Encoder) |
| `article_model` | [ncbi/MedCPT-Article-Encoder](https://huggingface.co/ncbi/MedCPT-Article-Encoder) |
| `ce_model` | [ncbi/MedCPT-Cross-Encoder](https://huggingface.co/ncbi/MedCPT-Cross-Encoder) |

Specify each local directory in `configs/local.yaml`. A directory must contain configuration, tokenizer assets and complete weights. Renaming an incomplete download does not repair it. Record the downloaded repository revision and file checksums. Gate/Reader/Generator all use the configured Llama checkpoint; do not silently substitute Llama 3.1 when comparing against a Llama 3 experiment.

Qwen3 requires a recent enough Transformers version; the inference environment targets 4.51.3. Qwen thinking is disabled in the wrapper. The external MedRAG adapter uses legacy sentence-transformers APIs in a separate environment. Verify that interpreter explicitly:

```bash
.venv/bin/python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
.venv-retrieval/bin/python -c "import sentence_transformers, faiss; print(sentence_transformers.__version__)"
```

Set `retrieval_python` to the **absolute executable path** of `.venv-retrieval/bin/python`. Do not install arbitrary latest retrieval packages into the inference environment to fix an interpreter-routing mistake.

## 2. Benchmark Format

The public entry point accepts the existing nested benchmark JSON:

```json
{
  "medqa": {
    "example-001": {
      "question": "Which cell type primarily transports oxygen in human blood?",
      "options": {"A": "Erythrocytes", "B": "Platelets", "C": "Neutrophils", "D": "Lymphocytes"},
      "answer": "A"
    }
  }
}
```

Supported top-level keys: `medqa`, `mmlu`, `pubmedqa`, `bioasq`. `--dataset mmlu_med` aliases `mmlu`. Question IDs must be unique, nonempty and stable. Benchmark IDs are sorted lexicographically; the internal `index` is assigned from that ordering. Evidence uses `source_key` and checks question/options/index consistency rather than relying only on line numbers. Preserve IDs and option order across experiments.

For a small custom input, `--questions file.jsonl` accepts one object per line with `id` (or `source_key`), `question`, `options`, and optional `answer`. The three bundled examples were written for this package and are not taken from a benchmark. A content-addressed prepared benchmark is written next to the run directory under `.ugap_inputs/`; it is an input conversion, not historical evidence.

For MCQ tasks, `options` maps 2-5 uppercase labels from A-E to text. The current Reader uses the existing option-letter JSON answer parser. For semantic tasks, preserve this adapter mapping:

```json
{"pubmedqa": {"options": {"A": "yes", "B": "no", "C": "maybe"}}}
{"bioasq": {"options": {"A": "yes", "B": "no"}}}
```

These two lines illustrate option mappings, not complete question files. The Gate scores the legal semantic labels (`yes/no/maybe` or `yes/no`) directly; it does not add a fourth option. Reader option predictions are mapped back to semantic labels. Gold `answer` may be the mapped option letter or legal semantic label. Only BioASQ yes/no questions are supported, not factoid/list tasks.

Acquire benchmark splits from their official distributors or a legitimately obtained, documented benchmark export. Preserve original split membership and IDs, and record conversion rules. PubMedQA evidence-provided versus question-only protocols are different: this runner uses the `question` field plus its own retrieved context. It does not automatically add dataset-provided abstracts. State the protocol explicitly when reporting results.

## 3. MedText Corpus

The default `MedText` corpus in this adapter is **textbooks + StatPearls**, following the external MedRAG corpus organization. It does not mean all PubMed abstracts. See [MedRAG data preparation](https://github.com/gzxiong/MedRAG) for the upstream format and obtain content under applicable terms.

Expected directory structure under `db_dir`:

```text
corpus/
  textbooks/
    chunk/*.jsonl
    index/ncbi/MedCPT-Article-Encoder/
      faiss.index
      metadatas.jsonl
  statpearls/
    chunk/*.jsonl
    index/ncbi/MedCPT-Article-Encoder/
      faiss.index
      metadatas.jsonl
```

Each chunk record should follow the MedRAG format, including `id`, `title`, and `content`. A typical upstream record also includes `contents` (title and body combined) for indexing. Use upstream preparation rather than inventing metadata offsets. FAISS entries, metadata rows, chunk filenames and chunk line offsets must refer to the **same corpus snapshot**. Copying only an index from another corpus version is not sufficient.

Set `medrag_repo` to an external checkout containing `src/utils.py`. The adapter expects `CustomizeSentenceTransformer` and `Retriever` interfaces, with `get_relevant_documents`, `index` and `metadatas`. Pin a compatible checkout and retain its license/attribution. Record it with:

```bash
git -C external/MedRAG rev-parse HEAD
```

If indexes are absent, prepare them explicitly after installing the retrieval environment and setting `article_model`:

```bash
python run.py --config configs/local.yaml --stage prepare_indexes
```

Index creation can be expensive. It is corpus preparation, not a dependency on previous experiment predictions. Existing indexes are reused, never silently regenerated by a standard `--stage all` run. The adapter checks index cardinality against metadata and rejects empty or mismatched indexes.

## 4. Reproducibility Record

Keep a local record of the effective config, benchmark split/version and SHA256, model revisions, MedRAG commit, corpus/chunk/index hashes, dependency versions and GPU/driver. For example:

```bash
.venv/bin/python -m pip freeze > inference-environment.txt
.venv-retrieval/bin/python -m pip freeze > retrieval-environment.txt
nvidia-smi > gpu-environment.txt
sha256sum data/benchmark.json > benchmark.sha256
```

The run manifest hashes release source and benchmark input; it is not a substitute for an independently archived checkpoint/corpus snapshot. Keep asset manifests privately if they contain licensed paths or data. Review them before public sharing. Do not put access tokens, model weights or private paths in the repository.

## 5. Common Failures

- `No module named sentence_transformers`: check `retrieval_python`; retrieval must execute in its own environment.
- `header too large` while loading safetensors: verify the checkpoint files are genuine, complete weights, not HTML/error pages or incomplete downloads.
- CUDA out of memory: stop unrelated jobs, confirm the configured device, and check input budgets. CPU offloading is not promised by this runner. Changing model precision/budgets changes the experimental configuration.
- Missing chunk/index: complete corpus/index preparation before inference. Do not substitute an unrelated corpus just to pass the check.
- Input/config/code changed: use a new output directory; do not delete the manifest to force incompatible cache reuse.
- Context-budget error: inspect saved prompt lengths and the configured limits. The strict Reader rejects oversize prompts rather than silently truncating away sources. Increase the supported budget or use a separately named budget configuration.

Do not count index construction or downloaded assets as an existing experiment cache when performing the fresh-run acceptance test. The test requires an empty **experiment output** directory; fixed models, corpus and indexes are expected prerequisites.
