from fastapi import FastAPI
from api.text_endpoints import router as text_router
from api.audio_endpoints import router as audio_router

app = FastAPI()

# Mount all endpoints
app.include_router(text_router, prefix="/text", tags=["Text Analysis"])
app.include_router(audio_router, prefix="/audio", tags=["Audio Analysis"])