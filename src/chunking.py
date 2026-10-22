import re
from dataclasses import dataclass
from pathlib import Path

from src import config

@dataclass
class Chunk:
    chunk_id: str 
    source: str 
    heading: str 
    text: str 

def _split_sections(text):
    """Yield (heading, body) pairs, splitting on markdown headings."""
    heading, buffer = "", []
    for line in text.splitlines():
        if re.match(r"^#{1,6}\s", line):
            body = "\n".join(buffer).strip()
            if body:
                yield heading, body
            heading, buffer = line.lstrip("#").strip(), []
        else:
            buffer.append(line)
    body = "\n".join(buffer).strip()
    if body:
        yield heading, body

def _units(body):
    """Break text into sentence-ish units (split on line breaks and . ! ?)."""
    units = []
    for para in re.split(r"\n+", body):
        para = para.strip()
        if not para:
            continue
        units.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip())
    return units

def _pack(units, size, overlap):
    """Pack whole sentences into chunks of up to `size` words,
    carrying the last ~`overlap` words of sentences into the next chunk."""
    chunks, current, count = [], [], 0
    for unit in units:
        n = len(unit.split())
        if n > size:  # one giant sentence: fall back to hard word splits
            if current:
                chunks.append(" ".join(current))
                current, count = [], 0
            words = unit.split()
            for i in range(0, len(words), size):
                chunks.append(" ".join(words[i:i + size]))
            continue
        if current and count + n > size:
            chunks.append(" ".join(current))
            carry, c = [], 0
            for u in reversed(current):
                w = len(u.split())
                if c + w > overlap:
                    break
                carry.insert(0, u)
                c += w
            current, count = carry, c
        current.append(unit)
        count += n
    if current:
        chunks.append(" ".join(current))
    return chunks

def chunk_file(path, size=None, overlap=None):
    size = size or config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    path = Path(path)
    text = _clean(path.read_text(encoding="utf-8-sig"))

    chunks, n = [], 0
    for heading, body in _split_sections(text):
        for piece in _pack(_units(body), size, overlap):
            full = f"{heading}\n\n{piece}" if heading else piece
            chunks.append(Chunk(f"{path.name}::{n}", path.name, heading, full))
            n += 1
    return chunks

def chunk_directory(directory, patterns=("*.md", "*.txt")):
    chunks = []
    for pattern in patterns:
        for path in sorted(Path(directory).rglob(pattern)):
            chunks.extend(chunk_file(path))
    return chunks

def _clean(text):
    text = re.sub(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", r"\1", text)      # **bold**
    text = re.sub(r"(?<!\*)\*(?=\S)(.+?)(?<=\S)\*(?!\*)", r"\1", text)  # *italic*
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)            # [text](url)
    return text