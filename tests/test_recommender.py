from src.recommender import Recommender, recommend_songs, score_song
from src.models import Song, UserProfile


def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    explanation = rec.explain_recommendation(user, rec.songs[0])

    assert isinstance(explanation, str)
    assert "Total score" in explanation


def test_retrieved_context_changes_song_score():
    user = {
        "favorite_genre": "ambient",
        "favorite_mood": "peaceful",
        "target_energy": 0.3,
        "likes_acoustic": True,
    }
    song = {
        "id": 6,
        "title": "Spacewalk Thoughts",
        "artist": "Orbit Bloom",
        "genre": "ambient",
        "mood": "chill",
        "energy": 0.28,
        "tempo_bpm": 60,
        "valence": 0.65,
        "danceability": 0.41,
        "acousticness": 0.92,
    }
    base_score, _ = score_song(user, song)
    rag_score, reasons = score_song(
        user,
        song,
        context={
            "genre_boosts": {"ambient": 0.5},
            "mood_boosts": {"chill": 0.4},
            "song_boosts": {"6": 0.3},
            "avoid_moods": [],
            "avoid_genres": [],
        },
    )

    assert rag_score > base_score
    assert any("retrieved guide" in reason for reason in reasons)


def test_recommend_songs_keeps_functional_api():
    songs = [
        {
            "id": 1,
            "title": "A",
            "artist": "B",
            "genre": "lofi",
            "mood": "focused",
            "energy": 0.4,
            "tempo_bpm": 80,
            "valence": 0.6,
            "danceability": 0.5,
            "acousticness": 0.8,
        }
    ]
    results = recommend_songs(
        {
            "favorite_genre": "lofi",
            "favorite_mood": "focused",
            "target_energy": 0.4,
            "likes_acoustic": True,
        },
        songs,
        k=1,
    )

    assert results[0][0]["title"] == "A"
    assert results[0][1] > 0
    assert isinstance(results[0][2], str)
