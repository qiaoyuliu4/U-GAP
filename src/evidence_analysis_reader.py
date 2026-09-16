"""U-GAP runtime components extracted from the research implementation."""

import re

from src.medrag_snippet_retriever import MedRAGSnippetRetriever

from src.selected_evidence_retriever import SelectedEvidenceRetriever

DATASET_ALIASES = {
    "medqa": "medqa",
    "medmcqa": "medmcqa",
    "mmlu": "mmlu",
    "mmlu_med": "mmlu",
    "mmlu-med": "mmlu",
    "medddx": "MedDDx",
    "pubmedqa": "pubmedqa",
    "bioasq": "bioasq",
}

def _clean(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()

def canonical_dataset_name(name):
    key = _clean(name).lower()
    if key not in DATASET_ALIASES:
        raise ValueError("Unsupported evidence-analysis dataset: %s" % name)
    return DATASET_ALIASES[key]

class FinalEvidenceLookup:
    """Align final evidence by source key and validate question identity."""

    def __init__(self, path, dataset, max_chars_per_evidence=1200):
        self.path = str(path)
        self.dataset = canonical_dataset_name(dataset).lower()
        self.reader = SelectedEvidenceRetriever(
            self.path,
            default_top_k=100000,
            max_chars_per_evidence=max_chars_per_evidence,
        )
        self.by_source_key = {}
        for record in self.reader.records:
            source_key = _clean(record.get("source_key"))
            if not source_key:
                continue
            if source_key in self.by_source_key:
                raise ValueError("Duplicate final-evidence source_key: %s" % source_key)
            self.by_source_key[source_key] = record

    def find(self, example):
        source_key = _clean(example.get("source_key"))
        record = self.by_source_key.get(source_key)
        if record is None:
            raise ValueError("Final evidence source_key not found: %s" % source_key)
        if int(record.get("index", -1)) != int(example["index"]):
            raise ValueError("Final evidence index mismatch for source_key %s" % source_key)
        record_dataset = record.get("dataset")
        if record_dataset and canonical_dataset_name(record_dataset).lower() != self.dataset:
            raise ValueError("Final evidence dataset mismatch at index %s" % example["index"])
        normalize = MedRAGSnippetRetriever.normalize_question
        if normalize(record.get("question", "")) != normalize(example["question"]):
            raise ValueError("Final evidence question mismatch at index %s" % example["index"])
        expected = {str(key).upper(): str(value) for key, value in example["options"].items()}
        actual = {str(key).upper(): str(value) for key, value in record.get("options", {}).items()}
        if actual and actual != expected:
            raise ValueError("Final evidence options mismatch at index %s" % example["index"])
        return record, "source_key"

    def format_record(self, record):
        return {
            "selected_text": self.reader.format_selected_text(record),
            "selected_kg": self.reader.format_selected_kg(record),
            "alignment_notes": self.reader.format_alignment_notes(record),
            "conflict_notes": self.reader.format_conflict_notes(record),
        }
