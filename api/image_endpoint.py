from fastapi import APIRouter, UploadFile, File, HTTPException
from schemas.image_schema import ImageAnalysisResponse
from controllers.image_controller import ImageController

router = APIRouter()
image_controller = ImageController(policy_docs="data/policy_docs")


@router.post("/analyze-image", response_model=ImageAnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    try:
        import tempfile
        import shutil

        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        result = image_controller.analyze_image(tmp_path)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return {
            "hate_speech": result.get("hate_speech", {}),
            "policies": result.get("policies", {}),
            "reasoning": result.get("reasoning", ""),
            "action": result.get("action", {}),
            "transcription": result.get("transcription", ""),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
