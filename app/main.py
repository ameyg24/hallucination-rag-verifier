from fastapi import FastAPI, HTTPException
from app.schemas import ( IngestTextRequest,
    IngestImageRequest,
    QueryRequest,
    QueryResponse,
    EvidenceItem,
)
from app.ingest import ingest_text, ingest_image_caption
from app.retrieval import HybridRetriever
from app.verify import verify_with_claims
from app.config import TOP_K, FAISS_WEIGHT, BM25_WEIGHT
from app.persist import save_store, load_store

app = FastAPI(title="AI Hallucination Detection & Multimodal RAG Verification Tool")

retriever = HybridRetriever()

# In-memory store
ALL_TEXTS = []
ALL_METAS = []

# Dedup guards
INGESTED_DOC_IDS = set()
INGESTED_IMAGE_IDS = set()


def _rebuild_dedup_sets():
    global INGESTED_DOC_IDS, INGESTED_IMAGE_IDS
    INGESTED_DOC_IDS = {met.get("doc_id") for met in ALL_METAS if met.get("doc_id")}
    INGESTED_IMAGE_IDS = {met.get("image_id") for met in ALL_METAS if met.get("image_id")}


@app.get("/health")
def health():
    return {"status": "ok", "indexed_items": len(ALL_TEXTS)}


@app.get("/stats")
def stats():
    doc_ids = {met.get("doc_id") for met in ALL_METAS if met.get("doc_id")}
    image_ids = {met.get("image_id") for met in ALL_METAS if met.get("image_id")}
    return {
        "total_indexed": len(ALL_TEXTS),
        "unique_doc_ids": len(doc_ids),
        "unique_image_ids": len(image_ids),
        "modalities": {
            "text": sum(1 for met in ALL_METAS if met.get("modality") == "text"),
            "image": sum(1 for met in ALL_METAS if met.get("modality") == "image"),
        },
        "weights": {"faiss": FAISS_WEIGHT, "bm25": BM25_WEIGHT},
        "top_k": TOP_K,
    }


@app.post("/reset")
def reset():
    global ALL_TEXTS, ALL_METAS, INGESTED_DOC_IDS, INGESTED_IMAGE_IDS
    ALL_TEXTS = []
    ALL_METAS = []
    INGESTED_DOC_IDS = set()
    INGESTED_IMAGE_IDS = set()
    retriever.build([], [])
    return {"status": "reset", "indexed_items": 0}


@app.post("/save_index")
def save_index():
    path = save_store(ALL_TEXTS, ALL_METAS)
    return {"status": "saved", "path": path, "indexed_items": len(ALL_TEXTS)}


@app.post("/load_index")
def load_index():
    global ALL_TEXTS, ALL_METAS
    texts, metas = load_store()
    ALL_TEXTS = texts
    ALL_METAS = metas
    _rebuild_dedup_sets()
    try:
        retriever.build(ALL_TEXTS, ALL_METAS)
    except ValueError as err:
        raise HTTPException(status_code=500, detail=f"invalid persisted index: {err}")
    return {"status": "loaded", "indexed_items": len(ALL_TEXTS)}


@app.post("/ingest/text")
def ingest_text_endpoint(req: IngestTextRequest):
    global ALL_TEXTS, ALL_METAS

    if req.doc_id in INGESTED_DOC_IDS:
        return {
            "ingested_chunks": 0,
            "total_indexed": len(ALL_TEXTS),
            "note": "doc_id already ingested",
        }

    chunks, metas = ingest_text(req.doc_id, req.text, req.metadata)
    if not chunks:
        return {
            "ingested_chunks": 0,
            "total_indexed": len(ALL_TEXTS),
            "note": "empty text produced no chunks",
        }

    ALL_TEXTS.extend(chunks)
    ALL_METAS.extend(metas)
    INGESTED_DOC_IDS.add(req.doc_id)

    retriever.build(ALL_TEXTS, ALL_METAS)
    return {"ingested_chunks": len(chunks), "total_indexed": len(ALL_TEXTS)}


@app.post("/ingest/image")
def ingest_image_endpoint(req: IngestImageRequest):
    global ALL_TEXTS, ALL_METAS

    if req.image_id in INGESTED_IMAGE_IDS:
        return {
            "ingested_images": 0,
            "total_indexed": len(ALL_TEXTS),
            "note": "image_id already ingested",
        }

    chunks, metas = ingest_image_caption(req.image_id, req.caption, req.metadata)
    if not chunks:
        return {
            "ingested_images": 0,
            "total_indexed": len(ALL_TEXTS),
            "note": "empty caption produced no chunks",
        }

    ALL_TEXTS.extend(chunks)
    ALL_METAS.extend(metas)
    INGESTED_IMAGE_IDS.add(req.image_id)

    retriever.build(ALL_TEXTS, ALL_METAS)
    return {"ingested_images": len(chunks), "total_indexed": len(ALL_TEXTS)}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    results = retriever.search(
        req.query,
        top_k=TOP_K,
        w_faiss=FAISS_WEIGHT,
        w_bm25=BM25_WEIGHT,
    )

    evidence_items = []
    for rec in results:
        evidence_items.append(
            EvidenceItem(
                source_id=rec["source_id"],
                modality=rec["modality"],
                content=rec["content"],
                metadata=rec["metadata"],
                score=rec["score"],
            )
        )

    evidence_dicts = [evi.model_dump() for evi in evidence_items]

    claim_text = req.claim or req.query
    verdict, answer, claim_results = verify_with_claims(
        query=req.query,
        draft_answer=claim_text,  # verify claim if provided
        evidence=evidence_dicts,
    )

    # For this "verification" mode, return a conservative answer
    if verdict == "supported":
        answer = "Supported by retrieved evidence."
    elif verdict == "uncertain":
        answer = "Partially supported, but evidence is not strong enough to be confident."
    else:
        answer = "I could not find supporting evidence in the retrieved sources."

    return QueryResponse(
        answer=answer,
        verdict=verdict,
        evidence=evidence_items,
        claims=claim_results,
        debug={
            "top_k": TOP_K,
            "weights": {"faiss": FAISS_WEIGHT, "bm25": BM25_WEIGHT},
            "indexed_items": len(ALL_TEXTS),
        },
    )
