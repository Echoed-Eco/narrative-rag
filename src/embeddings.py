from functools import lru_cache
from sentence_transformers import SentenceTransformer
from src import config


@lru_cache(maxsize=1)
def _model():
    return SentenceTransformer(config.EMBEDDING_MODEL)


def dimension():
    return _model().get_embedding_dimension()


def embed(texts, batch_size=32):
    """Return a (len(texts), dim) numpy array of unit-length vectors."""
    return _model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 50,
    )