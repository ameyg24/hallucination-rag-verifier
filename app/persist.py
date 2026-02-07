import json
from typing import Any, Dict, List, Tuple

from app.config import INDEX_DIR

STORE_PATH = INDEX_DIR / "store.json"


def save_store(text_items: List[str], meta_items: List[Dict[str, Any]]) -> str:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "text_items": text_items,
        "image_items": meta_items,
    }
    STORE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(STORE_PATH)


def load_store() -> Tuple[List[str], List[Dict[str, Any]]]:
    if not STORE_PATH.exists():
        return [], []
    payload = json.loads(STORE_PATH.read_text(encoding="utf-8"))
   
   
    # Backward compatibility
    text_items = payload.get("text_items")
    image_items = payload.get("image_items")
    if text_items is None and image_items is None:
        text_items = payload.get("all_texts", [])
        image_items = payload.get("all_metas", [])

    text_items = text_items or []
    image_items = image_items or []

    if not isinstance(text_items, list) or not all(isinstance(item, str) for item in text_items):
        raise ValueError("Invalid persisted store: text_items must be a list of strings")
    if not isinstance(image_items, list) or not all(isinstance(item, dict) for item in image_items):
        raise ValueError("Invalid persisted store: image_items must be a list of metadata objects")
    if len(text_items) != len(image_items):
        raise ValueError(
            "Invalid  persisted store: text_items and image_items must have the same length"
        )

    return text_items, image_items
