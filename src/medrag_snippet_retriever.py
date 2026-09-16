"""U-GAP runtime components extracted from the research implementation."""

import re

class MedRAGSnippetRetriever:

    """
    Lightweight reader for offline MedRAG-style retrieval results.

    Expected input can be JSONL or JSON. The loader is intentionally tolerant
    because different MedRAG scripts save snippets with slightly different keys.
    """

    SNIPPET_KEYS = (
        "snippets",
        "contexts",
        "ctxs",
        "docs",
        "documents",
        "retrieved_docs",
        "retrieved_snippets",
        "retrieved_contexts",
    )

    QUESTION_KEYS = ("question", "query", "input", "text")

    CONTENT_KEYS = ("content", "contents", "text", "passage", "abstract")

    TITLE_KEYS = ("title", "doc_title", "source", "corpus", "PMID", "id")

    @staticmethod
    def normalize_question(text):
        if text is None:
            return ""

        text = str(text).replace("\r", "\n").strip()
        lines = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if re.match(r"^[A-Ea-e]\s*[:.)]\s+", line):
                continue
            lines.append(line)

        normalized = " ".join(lines if lines else [text])
        normalized = normalized.lower()
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", "", normalized)
        return normalized.strip()
