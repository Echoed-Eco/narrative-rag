from collections import Counter
from src.chunking import chunk_directory

chunks = chunk_directory("data/sample")
print(f"{len(chunks)} chunks total\n")

for source, count in Counter(c.source for c in chunks).items():
    print(f"{source}: {count} chunks")

print("\n--- sample chunks (full text) ---")
step = max(len(chunks) // 4, 1)
for c in chunks[::step][:4]:
    print("-" * 40)
    print(c.chunk_id, "|", c.heading)
    print(c.text)