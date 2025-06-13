from pydantic import BaseModel
from typing import Any, Dict


class ImageAnalysisResponse(BaseModel):
    hate_speech: Dict[str, Any]
    policies: Any
    reasoning: str
    action: Dict[str, Any]
    transcription: str
