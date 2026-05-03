from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from .models import ProfileParseResult, UserProfile


ACTIVITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "study": {
        "keywords": ["study", "focus", "coding", "code", "work", "reading", "homework", "deep work"],
        "favorite_genre": "lofi",
        "favorite_mood": "focused",
        "target_energy": 0.38,
        "likes_acoustic": True,
        "desired_valence": 0.58,
        "target_danceability": 0.50,
        "diversity": 0.18,
    },
    "workout": {
        "keywords": ["workout", "gym", "run", "running", "lift", "training", "cardio", "hype"],
        "favorite_genre": "drum and bass",
        "favorite_mood": "driven",
        "target_energy": 0.90,
        "likes_acoustic": False,
        "desired_valence": 0.74,
        "target_danceability": 0.86,
        "diversity": 0.24,
    },
    "wind_down": {
        "keywords": ["sleep", "wind down", "bed", "bedtime", "quiet", "calm", "relax", "journal"],
        "favorite_genre": "ambient",
        "favorite_mood": "peaceful",
        "target_energy": 0.28,
        "likes_acoustic": True,
        "desired_valence": 0.56,
        "target_danceability": 0.36,
        "diversity": 0.20,
    },
    "party": {
        "keywords": ["party", "dance", "dancing", "hangout", "social", "club", "celebrate"],
        "favorite_genre": "house",
        "favorite_mood": "euphoric",
        "target_energy": 0.82,
        "likes_acoustic": False,
        "desired_valence": 0.84,
        "target_danceability": 0.88,
        "diversity": 0.35,
    },
    "commute": {
        "keywords": ["commute", "drive", "driving", "walk", "walking", "errands", "morning"],
        "favorite_genre": "synthwave",
        "favorite_mood": "moody",
        "target_energy": 0.70,
        "likes_acoustic": False,
        "desired_valence": 0.68,
        "target_danceability": 0.74,
        "diversity": 0.28,
    },
    "reflection": {
        "keywords": ["sad", "rainy", "breakup", "heartbreak", "reflective", "lonely", "nostalgic"],
        "favorite_genre": "folk",
        "favorite_mood": "melancholic",
        "target_energy": 0.40,
        "likes_acoustic": True,
        "desired_valence": 0.42,
        "target_danceability": 0.46,
        "diversity": 0.25,
    },
    "general": {
        "keywords": [],
        "favorite_genre": "indie pop",
        "favorite_mood": "happy",
        "target_energy": 0.62,
        "likes_acoustic": False,
        "desired_valence": 0.70,
        "target_danceability": 0.66,
        "diversity": 0.35,
    },
}


GENRE_ALIASES = {
    "dnb": "drum and bass",
    "drum n bass": "drum and bass",
    "hip-hop": "hip hop",
    "hiphop": "hip hop",
    "indie": "indie pop",
    "rnb": "r&b",
    "r and b": "r&b",
}


MOOD_ALIASES = {
    "happy": "happy",
    "joyful": "happy",
    "upbeat": "happy",
    "focused": "focused",
    "focus": "focused",
    "chill": "chill",
    "calm": "peaceful",
    "peaceful": "peaceful",
    "driven": "driven",
    "hype": "driven",
    "intense": "intense",
    "sad": "melancholic",
    "melancholy": "melancholic",
    "melancholic": "melancholic",
    "romantic": "romantic",
    "confident": "confident",
    "euphoric": "euphoric",
    "festive": "festive",
    "playful": "playful",
    "nostalgic": "nostalgic",
    "soulful": "soulful",
}


def _contains(text: str, phrase: str) -> bool:
    return phrase in text


def _first_label_match(text: str, labels: Iterable[str], aliases: Dict[str, str]) -> Optional[str]:
    sorted_labels = sorted(set(labels), key=len, reverse=True)
    for label in sorted_labels:
        if _contains(text, label.lower()):
            return label.lower()
    for alias, canonical in aliases.items():
        if _contains(text, alias):
            return canonical
    return None


class ProfileParser:
    def __init__(self, available_genres: Iterable[str], available_moods: Iterable[str]):
        self.available_genres = [genre.lower() for genre in available_genres]
        self.available_moods = [mood.lower() for mood in available_moods]

    def parse(
        self,
        query: str,
        explicit_preferences: Optional[Dict[str, Any]] = None,
    ) -> ProfileParseResult:
        explicit_preferences = explicit_preferences or {}
        text = query.strip().lower()
        activity, activity_score = self._choose_activity(text)
        preset = dict(ACTIVITY_PRESETS[activity])
        assumptions: List[str] = []
        missing_fields: List[str] = []

        if activity == "general":
            assumptions.append("No clear activity was detected, so the system used a balanced discovery profile.")
        else:
            assumptions.append(f"Detected activity: {activity}.")

        genre = _first_label_match(text, self.available_genres, GENRE_ALIASES)
        if genre:
            preset["favorite_genre"] = genre
        elif "favorite_genre" not in explicit_preferences and activity == "general":
            missing_fields.append("favorite_genre")

        mood = _first_label_match(text, self.available_moods, MOOD_ALIASES)
        if mood:
            preset["favorite_mood"] = mood
        elif "favorite_mood" not in explicit_preferences and activity == "general":
            missing_fields.append("favorite_mood")

        energy_override = self._energy_override(text)
        if energy_override is not None:
            preset["target_energy"] = energy_override
        elif "target_energy" not in explicit_preferences and activity == "general":
            missing_fields.append("target_energy")

        acoustic_override = self._acoustic_override(text)
        if acoustic_override is not None:
            preset["likes_acoustic"] = acoustic_override

        for key, value in explicit_preferences.items():
            if value is not None and value != "":
                preset[key] = value

        confidence = self._confidence(
            activity_score=activity_score,
            has_genre=bool(genre or explicit_preferences.get("favorite_genre")),
            has_mood=bool(mood or explicit_preferences.get("favorite_mood")),
            has_energy=energy_override is not None or "target_energy" in explicit_preferences or activity != "general",
            has_acoustic=acoustic_override is not None or "likes_acoustic" in explicit_preferences or activity != "general",
            query=text,
        )

        profile = UserProfile(
            favorite_genre=str(preset["favorite_genre"]).lower(),
            favorite_mood=str(preset["favorite_mood"]).lower(),
            target_energy=float(preset["target_energy"]),
            likes_acoustic=bool(preset["likes_acoustic"]),
            activity=activity,
            desired_valence=None
            if preset.get("desired_valence") is None
            else float(preset["desired_valence"]),
            target_danceability=None
            if preset.get("target_danceability") is None
            else float(preset["target_danceability"]),
            diversity=float(preset.get("diversity", 0.25)),
            novelty=float(preset.get("novelty", 0.35)),
            raw_query=query,
            assumptions=assumptions,
        )

        return ProfileParseResult(
            profile=profile,
            confidence=confidence,
            assumptions=assumptions,
            missing_fields=missing_fields,
        )

    def _choose_activity(self, text: str) -> Tuple[str, float]:
        best_activity = "general"
        best_score = 0
        for activity, preset in ACTIVITY_PRESETS.items():
            if activity == "general":
                continue
            score = sum(1 for keyword in preset["keywords"] if _contains(text, keyword))
            if score > best_score:
                best_activity = activity
                best_score = score
        if best_score == 0:
            return "general", 0.0
        return best_activity, min(1.0, 0.45 + best_score * 0.18)

    def _energy_override(self, text: str) -> Optional[float]:
        if any(term in text for term in ["high energy", "hype", "intense", "fast", "energetic"]):
            return 0.88
        if any(term in text for term in ["low energy", "quiet", "calm", "soft", "sleepy"]):
            return 0.30
        if any(term in text for term in ["medium energy", "balanced", "moderate"]):
            return 0.58
        return None

    def _acoustic_override(self, text: str) -> Optional[bool]:
        if any(term in text for term in ["non-acoustic", "not acoustic", "electronic", "synth", "beat-heavy"]):
            return False
        if any(term in text for term in ["acoustic", "unplugged", "guitar", "soft texture"]):
            return True
        return None

    def _confidence(
        self,
        activity_score: float,
        has_genre: bool,
        has_mood: bool,
        has_energy: bool,
        has_acoustic: bool,
        query: str,
    ) -> float:
        confidence = 0.28 + 0.25 * activity_score
        confidence += 0.14 if has_genre else 0.0
        confidence += 0.12 if has_mood else 0.0
        confidence += 0.10 if has_energy else 0.0
        confidence += 0.06 if has_acoustic else 0.0
        if len(query.split()) >= 6:
            confidence += 0.10
        return max(0.15, min(0.95, confidence))
