import sys

from src.rag import answer

question = " ".join(sys.argv[1:]) or "Tell me about Ilneva"
text, hits = answer(question)

print(text)
print("\nRetrieved:")
for chunk, score in hits:
    print(f"  {chunk.chunk_id} | {chunk.heading} (score {score:.3f})")