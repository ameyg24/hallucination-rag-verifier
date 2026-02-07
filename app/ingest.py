from typing import List, Dict, Any, Tuple
from app.utils import chunk_text

def ingest_text(doc_id: str, text: str, metadata: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
    chunks = chunk_text(text)
    metas = []
    for idx, _ in enumerate(chunks):
        metas.append({
            **(metadata or {}),
            "doc_id": doc_id,
            "chunk_id": idx,
            "modality": "text",
        })
    return chunks, metas

def ingest_image_caption(image_id: str, caption: str, metadata: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
    caption = (caption or "").strip()
    if not caption:
        return [], []

    chunks = [caption]
    metas = [{
        **(metadata or {}),
        "image_id": image_id,
        "chunk_id": 0,
        "modality": "image",
    }]
    
    return chunks, metas
