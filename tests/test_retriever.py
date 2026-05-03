from src.recommender import load_songs
from src.retriever import MusicRetriever
from src.models import UserProfile


def test_retriever_returns_study_guide_and_song_context():
    songs = load_songs("data/songs.csv")
    retriever = MusicRetriever(songs, "data/music_knowledge.json")
    profile = UserProfile(
        favorite_genre="lofi",
        favorite_mood="focused",
        target_energy=0.4,
        likes_acoustic=True,
        activity="study",
    )

    docs = retriever.retrieve("calm focus music for coding", profile, limit=6)
    context = retriever.build_context(docs)

    assert any(doc.kind == "knowledge" for doc in docs)
    assert any("Study" in doc.title for doc in docs)
    assert context["genre_boosts"]


def test_retriever_context_contains_song_boosts_for_specific_query():
    songs = load_songs("data/songs.csv")
    retriever = MusicRetriever(songs, "data/music_knowledge.json")
    profile = UserProfile(
        favorite_genre="lofi",
        favorite_mood="focused",
        target_energy=0.4,
        likes_acoustic=True,
        activity="study",
    )

    docs = retriever.retrieve("Focus Flow by LoRoom", profile, limit=5)
    context = retriever.build_context(docs)

    assert any(doc.kind == "song" for doc in docs)
    assert context["song_boosts"]
