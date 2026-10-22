from functools import lru_cache

from src import config
from src.chunking import chunk_directory
from src.embeddings import embed


@lru_cache(maxsize=1)
def _local_index():
    chunks = chunk_directory(config.DATA_DIR)
    return chunks, embed([c.text for c in chunks])


def _local(query, k):
    chunks, vectors = _local_index()
    scores = vectors @ embed([query])[0]
    top = scores.argsort()[::-1][:k]
    return [(chunks[i], float(scores[i])) for i in top]


def _azure(query, k, mode):
    from src import search_index  # imported here so local mode needs no Azure SDK
    return search_index.search(query, embed([query])[0], k=k, mode=mode)


def retrieve(query, k=None, mode=None):
    k = k or config.TOP_K
    if config.RETRIEVER == "azure":
        return _azure(query, k, mode or config.SEARCH_MODE)
    return _local(query, k)