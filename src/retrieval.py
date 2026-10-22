from functools import lru_cache

from src import config
from src.chunking import chunk_directory
from src.embeddings import embed


@lru_cache(maxsize=1)
def _index():
    chunks = chunk_directory(config.DATA_DIR)
    vectors = embed([c.text for c in chunks])
    return chunks, vectors


def retrieve(query, k=None):
    """Return the top-k (chunk, score) pairs for a query."""
    k = k or config.TOP_K
    chunks, vectors = _index()
    scores = vectors @ embed([query])[0]
    top = scores.argsort()[::-1][:k]
    return [(chunks[i], float(scores[i])) for i in top]