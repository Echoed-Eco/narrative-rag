import sys
from src import config
from src.chunking import chunk_directory
from src.embeddings import embed, dimension

query = " ".join(sys.argv[1:]) or "Who is the villain?"

chunks = chunk_directory("data/sample")
vectors = embed([c.text for c in chunks])
print(f"{len(chunks)} chunks embedded, dimension = {dimension()}\n")

scores = vectors @ embed([query])[0]
top = scores.argsort()[::-1][:config.TOP_K]

print(f"Query: {query}\n")
for rank, i in enumerate(top, 1):
    c = chunks[i]
    print(f"#{rank}  score={scores[i]:.3f}  {c.chunk_id} | {c.heading}")
    print("    " + c.text[:200].replace("\n", " ") + "...\n")