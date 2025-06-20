import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """
    Central configuration class for environment variables and model settings.
    """

    # DIAL API / Azure OpenAI settings from .env
    DIAL_API_KEY = os.getenv("DIAL_API_KEY")
    DIAL_API_VERSION = os.getenv("DIAL_API_VERSION")
    DIAL_API_ENDPOINT = os.getenv("DIAL_API_ENDPOINT", "https://ai-proxy.lab.epam.com")

    # Model names
    PRIMARY_MODEL_NAME = "gpt-4o-mini-2024-07-18"
    AUDIO_MODEL_NAME = "gpt-4o-mini-transcribe"
    EMBEDDING_MODEL_NAME = "text-embedding-3-small-1"

    # OPENAI API
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Reddit API
    REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
    REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
    REDDIT_USER_AGENT = os.getenv(
        "REDDIT_USER_AGENT", "RedditScrapingAgent/1.0 by u/IND_ROHAN"
    )

    # Qdrant settings
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

    @classmethod
    def validate_config(cls):
        """Validate that all required configuration variables are set"""
        required_vars = ["DIAL_API_KEY", "DIAL_API_VERSION", "DIAL_API_ENDPOINT"]
        missing_vars = [var for var in required_vars if not getattr(cls, var)]
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )
