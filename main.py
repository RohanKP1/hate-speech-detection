from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.text_endpoints import router as text_router
from api.audio_endpoints import router as audio_router

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all endpoints
app.include_router(text_router, prefix="/text", tags=["Text Analysis"])
app.include_router(audio_router, prefix="/audio", tags=["Audio Analysis"])