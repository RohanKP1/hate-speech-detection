from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class PolicySchema(BaseModel):
    content: str
    source: str
    relevance_score: Optional[float] = None
    rank: Optional[int] = None

class HateSpeechResultSchema(BaseModel):
    classification: str
    confidence: str
    reason: str

class ReasoningSchema(BaseModel):
    explanation: str

class ActionSchema(BaseModel):
    action: str
    severity: str
    reasoning: str

class AnalysisRequest(BaseModel):
    text: str = Field(default="", example="Your input text here.")

class AnalysisResponse(BaseModel):
    hate_speech: HateSpeechResultSchema
    policies: List[PolicySchema]
    reasoning: str
    action: ActionSchema

class ValidationRequest(BaseModel):
    text: str = Field(default=..., example="Your input text here.")

class ValidationResponse(BaseModel):
    validation: Dict[str, Any]