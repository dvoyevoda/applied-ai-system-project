from __future__ import annotations

import json
import math
import re
from pathlib import Path

from .models import KnowledgeItem, RetrievedDocument

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # pragma: no cover - fallback supports lightweight local runs
    TfidfVectorizer = None  # type: ignore[assignment]
    cosine_similarity = None  # type: ignore[assignment]


class KnowledgeRetriever:
    def __init__(self, knowledge_path: str | Path = "data/faq.json"):
        self.knowledge_path = Path(knowledge_path)
        self.items = self._load_items()
        self._documents = [self._document_text(item) for item in self.items]
        self._vectorizer = None
        self._matrix = None
        if TfidfVectorizer is not None and self._documents:
            self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            self._matrix = self._vectorizer.fit_transform(self._documents)

    def search(self, query: str, category: str | None = None, top_k: int = 3) -> list[RetrievedDocument]:
        if not query.strip() or not self.items:
            return []

        if self._vectorizer is not None and self._matrix is not None and cosine_similarity is not None:
            query_vector = self._vectorizer.transform([query])
            scores = cosine_similarity(query_vector, self._matrix).flatten().tolist()
        else:
            scores = [self._keyword_score(query, doc) for doc in self._documents]

        boosted = []
        for item, score in zip(self.items, scores):
            category_boost = 0.08 if category and item.category.lower() == category.lower() else 0.0
            boosted.append((item, float(score) + category_boost))

        ranked = sorted(boosted, key=lambda pair: pair[1], reverse=True)
        best = [RetrievedDocument(item=item, score=score) for item, score in ranked[:top_k] if score > 0]
        if not best and ranked:
            best = [RetrievedDocument(item=ranked[0][0], score=ranked[0][1])]
        return best

    def _load_items(self) -> list[KnowledgeItem]:
        if not self.knowledge_path.exists():
            return []
        with self.knowledge_path.open("r", encoding="utf-8") as file:
            raw = json.load(file)
        return [KnowledgeItem.from_dict(item) for item in raw]

    @staticmethod
    def _document_text(item: KnowledgeItem) -> str:
        tags = " ".join(item.tags)
        return f"{item.title} {item.category} {tags} {item.text}"

    @staticmethod
    def _keyword_score(query: str, document: str) -> float:
        query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        doc_terms = set(re.findall(r"[a-z0-9]+", document.lower()))
        if not query_terms or not doc_terms:
            return 0.0
        overlap = len(query_terms & doc_terms)
        return overlap / math.sqrt(len(query_terms) * len(doc_terms))
