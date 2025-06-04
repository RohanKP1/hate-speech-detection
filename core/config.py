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
    MODEL_NAME = "gpt-4o-mini-2024-07-18"

    # OPENAI API
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    @classmethod
    def validate_config(cls):
        """Validate that all required configuration variables are set"""
        required_vars = ['DIAL_API_KEY', 'DIAL_API_VERSION', 'DIAL_API_ENDPOINT']
        missing_vars = [var for var in required_vars if not getattr(cls, var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")