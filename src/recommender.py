from __future__ import annotations

import csv
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .models import Song, UserProfile


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "t", "yes", "y", "1", "acoustic"}:
            return True
        if lowered in {"false", "f", "no", "n", "0", "electronic"}:
            return False
    return default


def _text(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_user_prefs(user_prefs: Dict[str, Any] | UserProfile) -> Dict[str, Any]:
    if isinstance(user_prefs, UserProfile):
        user_prefs = user_prefs.to_dict()

    desired_valence = user_prefs.get("desired_valence")
    target_danceability = user_prefs.get("target_danceability")

    return {
        "favorite_genre": _text(user_prefs.get("favorite_genre", user_prefs.get("genre", ""))),
        "favorite_mood": _text(user_prefs.get("favorite_mood", user_prefs.get("mood", ""))),
        "target_energy": _clamp(
            _to_float(user_prefs.get("target_energy", user_prefs.get("energy", 0.5)), default=0.5)
        ),
        "likes_acoustic": _to_bool(
            user_prefs.get("likes_acoustic", user_prefs.get("prefer_acoustic", False)),
            default=False,
        ),
        "activity": _text(user_prefs.get("activity", "general")) or "general",
        "desired_valence": None if desired_valence is None else _clamp(_to_float(desired_valence, 0.5)),
        "target_danceability": None
        if target_danceability is None
        else _clamp(_to_float(target_danceability, 0.5)),
    }


def _normalize_song(song: Dict[str, Any] | Song) -> Dict[str, Any]:
    if isinstance(song, Song):
        return asdict(song)
    return song


def load_songs(csv_path: str) -> List[Dict[str, Any]]:
    songs: List[Dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            songs.append(
                {
                    "id": _to_int(row.get("id"), default=len(songs) + 1),
                    "title": str(row.get("title", "")).strip(),
                    "artist": str(row.get("artist", "")).strip(),
                    "genre": str(row.get("genre", "")).strip(),
                    "mood": str(row.get("mood", "")).strip(),
                    "energy": _clamp(_to_float(row.get("energy"), default=0.0)),
                    "tempo_bpm": _to_float(row.get("tempo_bpm"), default=0.0),
                    "valence": _clamp(_to_float(row.get("valence"), default=0.0)),
                    "danceability": _clamp(_to_float(row.get("danceability"), default=0.0)),
                    "acousticness": _clamp(_to_float(row.get("acousticness"), default=0.0)),
                }
            )
    return songs


def load_song_objects(csv_path: str) -> List[Song]:
    return [Song.from_dict(song) for song in load_songs(csv_path)]


def _context_boost(
    context: Optional[Dict[str, Any]], key: str, label: str, default: float = 0.0
) -> float:
    if not context:
        return default
    boosts = context.get(key, {})
    if not isinstance(boosts, dict):
        return default
    return _to_float(boosts.get(label, default), default=default)


def score_song(
    user_prefs: Dict[str, Any] | UserProfile,
    song: Dict[str, Any] | Song,
    context: Optional[Dict[str, Any]] = None,
) -> Tuple[float, List[str]]:
    prefs = _normalize_user_prefs(user_prefs)
    song_data = _normalize_song(song)
    song_genre = _text(song_data.get("genre", ""))
    song_mood = _text(song_data.get("mood", ""))
    song_id = str(song_data.get("id", ""))
    song_energy = _clamp(_to_float(song_data.get("energy"), default=0.0))
    song_acousticness = _clamp(_to_float(song_data.get("acousticness"), default=0.0))
    song_valence = _clamp(_to_float(song_data.get("valence"), default=0.0))
    song_danceability = _clamp(_to_float(song_data.get("danceability"), default=0.0))

    score = 0.0
    reasons: List[str] = []

    if prefs["favorite_genre"] and song_genre == prefs["favorite_genre"]:
        score += 2.0
        reasons.append("genre matches stated preference (+2.00)")

    if prefs["favorite_mood"] and song_mood == prefs["favorite_mood"]:
        score += 1.0
        reasons.append("mood matches stated preference (+1.00)")

    energy_points = _clamp(2.0 * (1.0 - abs(song_energy - prefs["target_energy"])), 0.0, 2.0)
    score += energy_points
    reasons.append(f"energy is close to target (+{energy_points:.2f})")

    if prefs["likes_acoustic"]:
        acoustic_points = song_acousticness
        reasons.append(f"acoustic preference adds +{acoustic_points:.2f}")
    else:
        acoustic_points = 1.0 - song_acousticness
        reasons.append(f"less-acoustic preference adds +{acoustic_points:.2f}")
    score += acoustic_points

    if prefs["desired_valence"] is not None:
        valence_points = 0.75 * (1.0 - abs(song_valence - prefs["desired_valence"]))
        valence_points = _clamp(valence_points, 0.0, 0.75)
        score += valence_points
        reasons.append(f"valence fits the requested tone (+{valence_points:.2f})")

    if prefs["target_danceability"] is not None:
        dance_points = 0.75 * (1.0 - abs(song_danceability - prefs["target_danceability"]))
        dance_points = _clamp(dance_points, 0.0, 0.75)
        score += dance_points
        reasons.append(f"danceability fits the activity (+{dance_points:.2f})")

    genre_context = _context_boost(context, "genre_boosts", song_genre)
    if genre_context:
        score += genre_context
        reasons.append(f"retrieved guide supports this genre (+{genre_context:.2f})")

    mood_context = _context_boost(context, "mood_boosts", song_mood)
    if mood_context:
        score += mood_context
        reasons.append(f"retrieved guide supports this mood (+{mood_context:.2f})")

    song_boost = _context_boost(context, "song_boosts", song_id)
    if song_boost:
        score += song_boost
        reasons.append(f"song matched retrieval query (+{song_boost:.2f})")

    avoid_moods = set(context.get("avoid_moods", [])) if context else set()
    avoid_genres = set(context.get("avoid_genres", [])) if context else set()
    if song_mood in avoid_moods:
        score -= 0.85
        reasons.append("guardrail penalty for mood mismatch (-0.85)")
    if song_genre in avoid_genres:
        score -= 0.85
        reasons.append("guardrail penalty for genre mismatch (-0.85)")

    return max(score, 0.0), reasons


def rank_songs(
    user_prefs: Dict[str, Any] | UserProfile,
    songs: Sequence[Dict[str, Any] | Song],
    context: Optional[Dict[str, Any]] = None,
) -> List[Tuple[Dict[str, Any], float, List[str]]]:
    ranked: List[Tuple[Dict[str, Any], float, List[str]]] = []
    for song in songs:
        song_data = _normalize_song(song)
        score, reasons = score_song(user_prefs, song_data, context=context)
        ranked.append((song_data, score, reasons))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def diversify_ranked(
    ranked: Sequence[Tuple[Dict[str, Any], float, List[str]]],
    k: int,
    diversity_strength: float = 0.25,
) -> List[Tuple[Dict[str, Any], float, List[str]]]:
    if k <= 0:
        return []
    if diversity_strength <= 0:
        return list(ranked[:k])

    selected: List[Tuple[Dict[str, Any], float, List[str]]] = []
    remaining = list(ranked)
    while remaining and len(selected) < k:
        best_index = 0
        best_adjusted = float("-inf")
        for index, (song, score, reasons) in enumerate(remaining):
            selected_genres = [_text(item[0].get("genre")) for item in selected]
            selected_moods = [_text(item[0].get("mood")) for item in selected]
            genre_overlap = selected_genres.count(_text(song.get("genre")))
            mood_overlap = selected_moods.count(_text(song.get("mood")))
            penalty = diversity_strength * (0.45 * genre_overlap + 0.25 * mood_overlap)
            adjusted_score = score - penalty
            if adjusted_score > best_adjusted:
                best_adjusted = adjusted_score
                best_index = index

        song, original_score, reasons = remaining.pop(best_index)
        final_reasons = list(reasons)
        if selected and best_adjusted < original_score:
            final_reasons.append(
                f"diversity reranker adjusted score from {original_score:.2f} to {best_adjusted:.2f}"
            )
        selected.append((song, best_adjusted, final_reasons))

    return selected


def recommend_songs(
    user_prefs: Dict[str, Any] | UserProfile,
    songs: List[Dict[str, Any]],
    k: int = 5,
    context: Optional[Dict[str, Any]] = None,
    diversity_strength: Optional[float] = None,
) -> List[Tuple[Dict[str, Any], float, str]]:
    if k <= 0:
        return []
    ranked = rank_songs(user_prefs, songs, context=context)
    if diversity_strength is None:
        diversity_strength = _to_float(
            user_prefs.diversity if isinstance(user_prefs, UserProfile) else user_prefs.get("diversity", 0.0),
            default=0.0,
        )
    selected = diversify_ranked(ranked, k=k, diversity_strength=diversity_strength)
    return [(song, score, "; ".join(reasons)) for song, score, reasons in selected]


class Recommender:
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        if k <= 0:
            return []
        ranked = rank_songs(user, self.songs)
        return [Song.from_dict(song) for song, _score, _reasons in ranked[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        score, reasons = score_song(user, song)
        return f"{song.title} by {song.artist}: {'; '.join(reasons)}. Total score: {score:.2f}"


def unique_labels(songs: Iterable[Dict[str, Any] | Song], field: str) -> List[str]:
    values = sorted({_text(_normalize_song(song).get(field, "")) for song in songs})
    return [value for value in values if value]
