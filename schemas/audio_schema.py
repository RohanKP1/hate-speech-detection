from pydantic import BaseModel
from typing import Any, Dict

class AudioAnalysisResponse(BaseModel):
    hate_speech: Dict[str, Any]
    policies: Any
    reasoning: str
    action: Dict[str, Any]

class AudioValidationResponse(BaseModel):
    validation: Dict[str, Any]