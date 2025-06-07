from fastapi import APIRouter, File, UploadFile, HTTPException
from schemas.audio_schema import AudioAnalysisResponse, AudioValidationResponse
from controllers.audio_controller import AudioController

router = APIRouter()
audio_controller = AudioController(policy_docs="data/policy_docs")


@router.post("/analyze-audio", response_model=AudioAnalysisResponse)
async def analyze_audio(file: UploadFile = File(...)):
    try:
        # Save uploaded file to a temporary location
        import tempfile
        import shutil

        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        result = audio_controller.analyze_audio(tmp_path)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return {
            "hate_speech": result.get("hate_speech", {}),
            "policies": result.get("policies", {}),
            "reasoning": result.get("reasoning", ""),
            "action": result.get("action", {}),
            "transcription": result.get("transcription", "")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate-audio", response_model=AudioValidationResponse)
async def validate_audio(file: UploadFile = File(...)):
    try:
        import tempfile
        import shutil

        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Transcribe and validate using the controller
        transcription = audio_controller.transcribe_audio(tmp_path)
        if isinstance(transcription, dict):
            if "error" in transcription:
                raise HTTPException(status_code=400, detail=transcription["error"])
            raise HTTPException(
                status_code=400, detail="Unexpected dictionary response"
            )
        validation_result = audio_controller.text_controller.validate_classification(
            transcription
        )
        if "error" in validation_result:
            raise HTTPException(status_code=400, detail=validation_result["error"])
        return {
            "validation": validation_result,
            "transcription": transcription,  # <-- Add the extracted text to the response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
