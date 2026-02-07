import os
from typing import List, Dict, Any, Optional

from openai import OpenAI

from app.config import OPENAI_MODEL


def _format_evidence(evidence: List[Dict[str, Any]], max_items: int = 6) -> str:
    lines = []
    for idx, item in enumerate(evidence[:max_items], start=1):
        source_id = item.get("source_id", "")
        modality = item.get("modality", "")
        content = item.get("content", "")
        lines.append(f"{idx}. [{modality}] {source_id}: {content}")
    return "\n".join(lines)


def _extract_text(resp: Any) -> str:
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text
    output = getattr(resp, "output", None)
    if not output:
        return ""
    # Best-effort parse of output items for message content
    for item in output:
        if getattr(item, "type", None) == "message":
            content = getattr(item, "content", None)
            if content:
                # content is a list of parts
                for part in content:
                    if getattr(part, "type", None) == "output_text":
                        return getattr(part, "text", "")
    return ""


def generate_answer_with_evidence(
    query: str,
    evidence: List[Dict[str, Any]],
    prompt: str,
    model: Optional[str] = None,
) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI()
    model_name = model or OPENAI_MODEL
    evidence_block = _format_evidence(evidence)

    instructions = prompt.strip()
    user_input = (
        f"Claim: {query}\n\n"
        f"Evidence:\n{evidence_block}\n\n"
        "Return a concise answer grounded  only in the evidence."
    )

    resp = client.responses.create(
        model=model_name,
        instructions=instructions,
        input=user_input,
    )
    text = _extract_text(resp)
    if not text:
        raise RuntimeError("OpenAI response had no text output")
    return text.strip()
