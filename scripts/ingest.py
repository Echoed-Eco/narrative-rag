from src import config
from src.chunking import chunk_directory
from src.embeddings import embed, dimension
from src import search_index

chunks = chunk_directory(config.DATA_DIR)
print(f"{len(chunks)} chunks")

vectors = embed([c.text for c in chunks])
search_index.create_index(dimension())
n = search_index.upload(chunks, vectors)
print(f"Uploaded {n} documents to '{config.SEARCH_INDEX_NAME}'")