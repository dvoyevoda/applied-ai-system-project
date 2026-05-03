from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import RetrievedDocument, UserProfile


def load_knowledge_records(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as knowledge_file:
        return json.load(knowledge_file)


class MusicRetriever:
    def __init__(self, songs: Sequence[Dict[str, Any]], knowledge_path: str | Path):
        self.songs = list(songs)
        self.knowledge_records = load_knowledge_records(knowledge_path)
        self.documents = self._build_documents()
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([doc["search_text"] for doc in self.documents])

    def retrieve(
        self,
        query: str,
        profile: UserProfile,
        limit: int = 8,
    ) -> List[RetrievedDocument]:
        query_text = self._query_text(query, profile)
        query_vector = self.vectorizer.transform([query_text])
        scores = cosine_similarity(query_vector, self.matrix).flatten()
        ranked_indexes = scores.argsort()[::-1]

        docs: List[RetrievedDocument] = []
        for index in ranked_indexes:
            score = float(scores[index])
            doc = self.documents[int(index)]
            if score <= 0.0 and len(docs) >= 3:
                continue
            docs.append(
                RetrievedDocument(
                    id=doc["id"],
                    title=doc["title"],
                    kind=doc["kind"],
                    text=doc["text"],
                    score=score,
                    metadata=dict(doc["metadata"]),
                )
            )
            if len(docs) >= limit:
                break

        if not any(doc.kind == "knowledge" for doc in docs):
            docs.extend(self._fallback_knowledge(profile.activity))
        return docs[:limit]

    def build_context(self, documents: Sequence[RetrievedDocument]) -> Dict[str, Any]:
        context: Dict[str, Any] = {
            "genre_boosts": {},
            "mood_boosts": {},
            "song_boosts": {},
            "avoid_moods": set(),
            "avoid_genres": set(),
            "evidence_titles": [],
        }

        for doc in documents:
            context["evidence_titles"].append(doc.title)
            normalized_score = min(1.0, max(0.05, doc.score))
            if doc.kind == "song":
                song_id = str(doc.metadata.get("song_id", ""))
                context["song_boosts"][song_id] = max(
                    context["song_boosts"].get(song_id, 0.0),
                    1.10 * normalized_score,
                )
                continue

            for genre in doc.metadata.get("boost_genres", []):
                genre = str(genre).lower()
                context["genre_boosts"][genre] = context["genre_boosts"].get(genre, 0.0) + 0.36
            for mood in doc.metadata.get("boost_moods", []):
                mood = str(mood).lower()
                context["mood_boosts"][mood] = context["mood_boosts"].get(mood, 0.0) + 0.30
            for mood in doc.metadata.get("avoid_moods", []):
                context["avoid_moods"].add(str(mood).lower())
            for genre in doc.metadata.get("avoid_genres", []):
                context["avoid_genres"].add(str(genre).lower())

        context["avoid_moods"] = sorted(context["avoid_moods"])
        context["avoid_genres"] = sorted(context["avoid_genres"])
        return context

    def _build_documents(self) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        for song in self.songs:
            energy_band = self._energy_band(float(song["energy"]))
            text = (
                f"{song['title']} by {song['artist']} is a {song['genre']} song with a "
                f"{song['mood']} mood, {energy_band} energy, tempo {song['tempo_bpm']} bpm, "
                f"valence {song['valence']}, danceability {song['danceability']}, and "
                f"acousticness {song['acousticness']}."
            )
            docs.append(
                {
                    "id": f"song.{song['id']}",
                    "title": f"{song['title']} by {song['artist']}",
                    "kind": "song",
                    "text": text,
                    "search_text": text,
                    "metadata": {
                        "song_id": str(song["id"]),
                        "genre": str(song["genre"]).lower(),
                        "mood": str(song["mood"]).lower(),
                    },
                }
            )

        for record in self.knowledge_records:
            text = f"{record['title']}. {record['content']}"
            metadata = {key: value for key, value in record.items() if key not in {"id", "title", "content"}}
            docs.append(
                {
                    "id": str(record["id"]),
                    "title": str(record["title"]),
                    "kind": "knowledge",
                    "text": str(record["content"]),
                    "search_text": text,
                    "metadata": metadata,
                }
            )
        return docs

    def _fallback_knowledge(self, activity: str) -> List[RetrievedDocument]:
        matches = [
            record for record in self.knowledge_records if record.get("activity") in {activity, "general"}
        ]
        fallback_docs = []
        for record in matches[:2]:
            metadata = {key: value for key, value in record.items() if key not in {"id", "title", "content"}}
            fallback_docs.append(
                RetrievedDocument(
                    id=str(record["id"]),
                    title=str(record["title"]),
                    kind="knowledge",
                    text=str(record["content"]),
                    score=0.05,
                    metadata=metadata,
                )
            )
        return fallback_docs

    def _query_text(self, query: str, profile: UserProfile) -> str:
        return " ".join(
            [
                query,
                profile.activity,
                profile.favorite_genre,
                profile.favorite_mood,
                "acoustic" if profile.likes_acoustic else "electronic beat",
                f"energy {profile.target_energy:.2f}",
            ]
        )

    def _energy_band(self, value: float) -> str:
        if value >= 0.75:
            return "high"
        if value <= 0.40:
            return "low"
        return "medium"
