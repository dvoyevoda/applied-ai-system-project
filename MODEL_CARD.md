# Model Card: MoodMap Music Recommender

## Model Name

MoodMap Music Recommender

## Intended Use

MoodMap recommends songs from a small classroom catalog based on a user's stated activity, genre, mood, energy, and acoustic preferences. It is intended for a CodePath applied AI project demo, not for production music personalization.

## How It Works

The system parses a natural-language music request into a structured taste profile. When an OpenAI API key is provided, an external LLM refines that taste profile and later writes grounded recommendation explanations from retrieved evidence. It retrieves relevant song and music-guide documents with TF-IDF, then uses the original Module 3 scoring formula plus retrieval boosts to rank songs. A diversity reranker and self-check layer produce the final output with confidence scores and guardrail flags.

## Data

The song catalog is `data/songs.csv`, a 28-song synthetic dataset with title, artist, genre, mood, energy, tempo, valence, danceability, and acousticness. The retrieval documents are in `data/music_knowledge.json` and cover activities like study, workout, wind-down, party, commute, and reflective listening.

## Strengths

- Transparent scoring and explanations.
- Uses an external LLM when an API key is provided, with a local fallback when no key is available.
- Uses retrieved context to meaningfully change recommendations.
- Includes confidence scoring, guardrails, logs, unit tests, and evaluation cases.

## Limitations

- Small catalog means limited discovery.
- Metadata quality strongly affects results.
- Rule-based parsing can miss unusual phrasing.
- Confidence is a reliability signal, not a guarantee of user satisfaction.

## Responsible Use

MoodMap should present recommendations as suggestions based on stated preferences. It should not infer sensitive traits, diagnose emotions, guarantee productivity, or replace human judgment for sensitive wellbeing requests.
