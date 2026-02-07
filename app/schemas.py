from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

class IngestTextRequest(BaseModel):
    doc_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class IngestImageRequest(BaseModel):
    image_id: str = Field(..., min_length=1)
    caption: str = Field(..., min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EvidenceItem(BaseModel):
    source_id: str
    modality: Literal["text", "image"]
    content: str
    metadata: Dict[str, Any]
    score: float = Field(..., ge=0.0, le=1.0)

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    claim: Optional[str] = None

class ClaimResult(BaseModel):
    claim: str
    verdict: Literal["supported", "unsupported", "uncertain"]
    citations: List[str]

class QueryResponse(BaseModel):
    answer: str
    verdict: Literal["supported", "unsupported", "uncertain"]
    evidence: List[EvidenceItem]
    claims: Optional[List[ClaimResult]] = None
    debug: Optional[Dict[str, Any]] = None
