from pydantic import BaseModel
from typing import Any, Dict


class AudioAnalysisResponse(BaseModel):
    hate_speech: Dict[str, Any]
    policies: Any
    reasoning: str
    action: Dict[str, Any]
    transcription: str


class AudioValidationResponse(BaseModel):
    validation: Dict[str, Any]
    transcription: str
