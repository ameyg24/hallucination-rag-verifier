from typing import List, Tuple, Dict, Any
import re


from app.config import VERIFIER_SUPPORTED_MIN_OVERLAP, VERIFIER_UNCERTAIN_MIN_OVERLAP
# VERIFY_VERSION = "negation_plural_fix_1"


def _norm(txt: str) -> str:
    return re.sub(r"\s+", " ", txt.strip().lower())


def _tokens(txt: str) -> List[str]:
    txt = _norm(txt)
    txt = re.sub(r"[^a-z0-9\s]", " ", txt)
    parts = [part for part in txt.split() if part]
    return [part for part in parts if len(part) >= 3]


def _overlap(left_txt: str, right_txt: str) -> int:
    left_tokens = set(_tokens(left_txt))
    right_tokens = set(_tokens(right_txt))
    return len(left_tokens.intersection(right_tokens))


def _contains_normalized_phrase(needle: str, haystack: str) -> bool:
    needle_norm = re.sub(r"[^a-z0-9\s]", " ", _norm(needle))
    haystack_norm = re.sub(r"[^a-z0-9\s]", " ", _norm(haystack))
    needle_norm = re.sub(r"\s+", " ", needle_norm).strip()
    haystack_norm = re.sub(r"\s+", " ", haystack_norm).strip()
    return bool(needle_norm) and needle_norm in haystack_norm


def _split_claims(text: str, max_claims: int = 3) -> List[str]:
    parts = [part.strip() for part in re.split(r"[.;\n]+", text) if part.strip()]
    return parts[:max_claims] if parts else []


def _relation_triplet(txt: str):
    norm_txt = _norm(txt)
    subjects = ["faiss", "bm25", "rag", "hybrid retrieval"]
    norm_txt = norm_txt.replace("hybrid-retrieval", "hybrid retrieval")

    for subj in subjects:
        if norm_txt.startswith(subj + " "):
            rest = norm_txt[len(subj) + 1 :]
            for pred in ["supports", "does", "is", "uses", "combines", "stores", "store"]:
                if rest.startswith(pred + " "):
                    obj = rest[len(pred) + 1 :].strip()
                    return subj, pred, obj
    return None, None, None


def _capability_tags(obj_phrase: str) -> List[str]:
    obj_norm = _norm(obj_phrase)
    tags = []
    if "keyword" in obj_norm or "ranking" in obj_norm or "tf" in obj_norm or "idf" in obj_norm:
        tags.append("keyword_ranking")
    if "similarity" in obj_norm or "vector" in obj_norm or "dense" in obj_norm or "nearest neighbor" in obj_norm:
        tags.append("vector_similarity")
    if "embed" in obj_norm or "embedding" in obj_norm or "embeddings" in obj_norm:
        tags.append("embeddings")
    if "store" in obj_norm or "stores" in obj_norm or "storage" in obj_norm:
        tags.append("stores")
    if "deprecated" in obj_norm:
        tags.append("deprecated")
    return tags


def _has_embedding_word(txt: str) -> bool:
    txt = _norm(txt)
    return ("embedding" in txt) or ("embeddings" in txt) or ("embed" in txt)


def _negation_conflict(claim: str, evidence_text: str) -> bool:
    """
    If evidence explicitly negates the claim, return True.
    """
    claim_norm = _norm(claim)
    evidence_norm = _norm(evidence_text)

    # BM25 embeddings storage contradiction (handles embedding vs embeddings)
    if "bm25" in claim_norm and "store" in claim_norm and _has_embedding_word(claim_norm):
        if "bm25 does not store" in evidence_norm and _has_embedding_word(evidence_norm):
            return True
        
        if "bm25 doesn't store" in evidence_norm and _has_embedding_word(evidence_norm):
            return True

    # FAISS keyword ranking contradiction
    if "faiss" in claim_norm and ("keyword" in claim_norm and "ranking" in claim_norm):
        if ("faiss is not" in evidence_norm or "faiss isn't" in evidence_norm) and ("keyword" in evidence_norm and "ranking" in evidence_norm):
            return True

    # Generic contradiction: "does not" + key terms that are present in claim
    if "does not" in evidence_norm:
        if _has_embedding_word(claim_norm) and _has_embedding_word(evidence_norm) and "store" in claim_norm and "store" in evidence_norm:
            return True

    return False


def verify_with_claims(
    query: str,
    draft_answer: str,
    evidence: List[Dict[str, Any]],
    supported_min_overlap: int = VERIFIER_SUPPORTED_MIN_OVERLAP,
    uncertain_min_overlap: int = VERIFIER_UNCERTAIN_MIN_OVERLAP,
) -> Tuple[str, str, List[Dict[str, Any]]]:
    if not evidence:
        return "unsupported", "I could not find supporting evidence in the retrieved sources.", []

    if not draft_answer.strip():
        draft_answer = query.strip()

    claims = _split_claims(draft_answer, max_claims=3)
    if not claims:
        claims = [draft_answer.strip()]

    claim_results = []
    supported = 0
    uncertain = 0
    max_evidence_per_claim = 3

    for claim in claims:
        claim_norm = _norm(claim)
        claim_subj, claim_pred, claim_obj = _relation_triplet(claim_norm)
        claim_tags = _capability_tags(claim_obj or "")

        scored: List[Tuple[int, int, str]] = []

        for idx, ev_item in enumerate(evidence):
            ev_text = ev_item.get("content", "")
            score = _overlap(claim, ev_text)
            scored.append((score, idx, ev_text))

        scored.sort(key=lambda item: item[0], reverse=True)
        top = scored[:max_evidence_per_claim]
        best_score, best_idx, best_ev = (top[0] if top else (0, -1, ""))
        exact_match_idx = next(
            (
                idx
                for _, idx, ev_text in top
                if ev_text and _contains_normalized_phrase(claim, ev_text)
            ),
            -1,
        )

        # Claims need exact evidence containment
        if exact_match_idx >= 0:
            verdict = "supported"
            citations = [evidence[exact_match_idx].get("source_id", "")]
        elif best_score >= uncertain_min_overlap:
            verdict = "uncertain"
            citations = [evidence[best_idx].get("source_id", "")] if best_idx >= 0 else []
        else:
            verdict = "unsupported"
            citations = []

        # Negation contradiction across top evidence overrides  overlap
        if verdict in ("supported", "uncertain"):
            has_contradiction = any(_negation_conflict(claim, ev_text) for _, _, ev_text in top if ev_text)
            if has_contradiction:
                verdict = "unsupported"
                citations = []

        # Subject/capability mismattch rule
        if verdict in ("supported", "uncertain") and claim_subj in ("faiss", "bm25") and claim_tags:
            for _, _, ev_text in top:
                ev_subj, ev_pred, ev_obj = _relation_triplet(ev_text or "")
                ev_tags = _capability_tags(ev_obj or "")
                if ev_subj in ("faiss", "bm25") and ev_subj != claim_subj:
                    if any(tag in ev_tags for tag in claim_tags):
                        verdict = "unsupported"
                        citations = []
                        break

        # "deprecated" handling rule
        if "deprecated" in claim_norm and verdict in ("supported", "uncertain"):
            if not any("deprecated" in _norm(ev_text or "") for _, _, ev_text in top):
                verdict = "unsupported"
                citations = []

        if verdict == "supported":
            supported += 1
        elif verdict == "uncertain":
            uncertain += 1

        claim_results.append(
            {"claim": claim, "verdict": verdict, "citations": [cite for cite in citations if cite]}
        )

    if supported == len(claim_results):
        overall = "supported"
        final_answer = "Supported by retrieved evidence."
    elif supported == 0 and uncertain == 0:
        overall = "unsupported"
        # If retrieved evidence, it likely contradicts or doesn't support the claim
        if evidence:
            final_answer = "The retrieved evidence does not support this claim."
        else:
            final_answer = "I could not find supporting evidence in the retrieved sources."
    else:
        overall = "uncertain"
        final_answer = "Partially supported, but evidence is not strong enough to be confident."

    return overall, final_answer, claim_results
