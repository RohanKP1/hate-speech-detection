from fastapi import APIRouter, HTTPException
from schemas.text_schema import AnalysisRequest, AnalysisResponse, ValidationRequest, ValidationResponse
from controllers.text_controller import TextController

router = APIRouter()
controller = TextController(policy_docs="data/policy_docs")

@router.post("/analyze", response_model=AnalysisResponse)
def analyze_text(request: AnalysisRequest):
    result = controller.analyze_text(request.text)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return {
        "hate_speech": result["hate_speech"],
        "policies": result["policies"],
        "reasoning": result["reasoning"],
        "action": result["action"]
    }

@router.post("/validate", response_model=ValidationResponse)
def validate_text(request: ValidationRequest):
    result = controller.validate_classification(request.text)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return {"validation": result}