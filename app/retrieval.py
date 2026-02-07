from typing import List, Dict, Any
import numpy as np
import faiss
from rank_bm25 import BM25Okapi

from app.utils import clean_text, simple_tokens

class HybridRetriever:
    def __init__(self):
        self.texts: List[str] = []
        self.metas: List[Dict[str, Any]] = []

        self.bm25 = None
        # self.bm25_tokens = None

        self.vectorizer_vocab = {}
        self.faiss_index = None
        self.dim = 0
        # self.doc_vecs = None

    def _reset_indices(self):
        self.bm25 = None
        self.bm25_tokens = None
        self.vectorizer_vocab = {}
        self.faiss_index = None
        self.dim = 0
        self.doc_vecs = None

    def _build_tfidf_vocab(self, texts: List[str], max_features: int = 50000):
        freq = {}
        for txt in texts:
            for tok in set(simple_tokens(txt)):
                freq[tok] = freq.get(tok, 0) + 1
        items = sorted(freq.items(), key=lambda itm: itm[1], reverse=True)[:max_features]
        self.vectorizer_vocab = {wrd: idx for idx, (wrd, _) in enumerate(items)}
        self.dim = len(self.vectorizer_vocab)

    def _embed(self, texts: List[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype="float32")
        for row, txt in enumerate(texts):
            toks = simple_tokens(txt)
            for tok in toks:
                col = self.vectorizer_vocab.get(tok)
                if col is not None:
                    mat[row, col] += 1.0
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
        mat = mat / norms
        return mat

    def build(self, texts: List[str], metas: List[Dict[str, Any]]):
        if len(texts) != len(metas):
            raise ValueError("texts and metas must have the same length")

        self._reset_indices()
        self.texts = texts
        self.metas = metas

        if not texts:
            return

        # BM25
        tokenized = [simple_tokens(txt) for txt in texts]
        self.bm25_tokens = tokenized
        self.bm25 = BM25Okapi(tokenized)

        # FAISS
        self._build_tfidf_vocab(texts)
        vecs = self._embed(texts)
        self.doc_vecs = vecs

        self.faiss_index = faiss.IndexFlatIP(self.dim)
        self.faiss_index.add(vecs)

    def search(self, query: str, top_k: int = 6, w_faiss: float = 0.6, w_bm25: float = 0.4):
        qry = clean_text(query)
        if not qry or self.faiss_index is None or self.bm25 is None:
            return []

        qvec = self._embed([qry])
        if self.dim == 0 or float(np.linalg.norm(qvec)) < 1e-9:
            return []

        faiss_scores, faiss_ids = self.faiss_index.search(qvec, top_k * 5)
        faiss_scores = faiss_scores[0].tolist()
        faiss_ids = faiss_ids[0].tolist()

        # BM25
        bm_query_tokens = simple_tokens(qry)
        if not bm_query_tokens:
            return []
        bm_scores = self.bm25.get_scores(bm_query_tokens).tolist()
        if not any(scr > 0 for scr in bm_scores):
            return []

        def norm(scores):
            smin, smax = min(scores), max(scores)
            if smax - smin < 1e-9:
                return [0.0 for _ in scores]
            return [(scr - smin) / (smax - smin) for scr in scores]

        bm_norm = norm(bm_scores)

        bm_top = sorted(range(len(bm_scores)), key=lambda idx: bm_scores[idx], reverse=True)[: top_k * 5]
        candidates = set([idx for idx in faiss_ids if idx != -1]) | set(bm_top)
        merged = []
        faiss_map = {idx: scr for idx, scr in zip(faiss_ids, faiss_scores) if idx != -1}
        for idx in candidates:
            fscore = float(faiss_map.get(idx, 0.0))
            fscore = float(max(0.0, min(1.0, fscore)))
            score = w_faiss * fscore + w_bm25 * bm_norm[idx]
            merged.append((idx, score))

        merged.sort(key=lambda tup: tup[1], reverse=True)

        out = []
        for idx, score in merged[:top_k]:
            meta = self.metas[idx]
            modality = meta.get("modality", "text")
            source_id = ""
            if modality == "text":
                source_id = f'{meta.get("doc_id")}#{meta.get("chunk_id")}'
            else:
                source_id = f'{meta.get("image_id")}#caption'

            out.append({
                "source_id": source_id,
                "modality": modality,
                "content": self.texts[idx],
                "metadata": meta,
                "score": float(score),
            })
        return out
