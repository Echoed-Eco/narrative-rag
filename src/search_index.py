import re

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery

from src import config
from src.chunking import Chunk


def _cred():
    if not config.SEARCH_ENDPOINT or not config.SEARCH_API_KEY:
        raise RuntimeError("Set SEARCH_ENDPOINT and SEARCH_API_KEY in .env")
    return AzureKeyCredential(config.SEARCH_API_KEY)


def _index_client():
    return SearchIndexClient(config.SEARCH_ENDPOINT, _cred())


def _search_client():
    return SearchClient(config.SEARCH_ENDPOINT, config.SEARCH_INDEX_NAME, _cred())


def _safe_key(chunk_id):
    # Azure keys only allow letters, digits, _ - =   (no "." or ":")
    return re.sub(r"[^A-Za-z0-9_\-=]", "_", chunk_id)


def create_index(dimensions):
    """Drop and recreate the index so it always matches the current chunks."""
    client = _index_client()
    try:
        client.delete_index(config.SEARCH_INDEX_NAME)
    except ResourceNotFoundError:
        pass

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="chunk_id", type=SearchFieldDataType.String),
        SimpleField(name="source", type=SearchFieldDataType.String,
                    filterable=True, facetable=True),
        SearchableField(name="heading", type=SearchFieldDataType.String),
        SearchableField(name="text", type=SearchFieldDataType.String),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=dimensions,
            vector_search_profile_name="vec-profile",
        ),
    ]
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
        profiles=[VectorSearchProfile(name="vec-profile",
                                      algorithm_configuration_name="hnsw")],
    )
    client.create_index(
        SearchIndex(name=config.SEARCH_INDEX_NAME, fields=fields,
                    vector_search=vector_search)
    )


def upload(chunks, vectors, batch_size=100):
    client = _search_client()
    docs = [
        {
            "id": _safe_key(c.chunk_id),
            "chunk_id": c.chunk_id,
            "source": c.source,
            "heading": c.heading,
            "text": c.text,
            "embedding": v.tolist(),
        }
        for c, v in zip(chunks, vectors)
    ]
    for i in range(0, len(docs), batch_size):
        client.upload_documents(documents=docs[i:i + batch_size])
    return len(docs)


def search(query_text, query_vector, k=5, mode="hybrid"):
    """Return [(Chunk, score)]. mode='vector' is vector-only; 'hybrid' adds keyword search."""
    vq = VectorizedQuery(vector=query_vector.tolist(),
                         k_nearest_neighbors=k, fields="embedding")
    results = _search_client().search(
        search_text=query_text if mode == "hybrid" else None,
        vector_queries=[vq],
        top=k,
    )
    return [
        (Chunk(r["chunk_id"], r["source"], r["heading"], r["text"]), r["@search.score"])
        for r in results
    ]