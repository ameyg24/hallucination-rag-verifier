import re
from typing import List
from app.config import CHUNK_SIZE, CHUNK_OVERLAP

def clean_text(txt: str) -> str:
    txt = txt.replace("\u00a0", " ")
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def chunk_text(text: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(text_len, start + CHUNK_SIZE)
        chunks.append(text[start:end])
        if end >= text_len:
            break
        start = max(0, end - CHUNK_OVERLAP)
    return chunks

def simple_tokens(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", clean_text(text).lower())
