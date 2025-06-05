import os
import requests
from typing import Dict
from utils.custom_logger import CustomLogger

logger = CustomLogger("HateSpeechListener")

class HateSpeechListener:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        logger.info("HateSpeechListener initialized")

    def analyze_audio(self, file_path: str) -> Dict:
        endpoint = f"{self.base_url}/audio/analyze-audio"
        return self._send_audio(file_path, endpoint)

    def validate_audio(self, file_path: str) -> Dict:
        endpoint = f"{self.base_url}/audio/validate-audio"
        return self._send_audio(file_path, endpoint)

    def analyze_text(self, text: str) -> Dict:
        endpoint = f"{self.base_url}/text/analyze"
        payload = {"text": text}
        try:
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Text analysis error: {str(e)}")
            return {"error": str(e)}

    def validate_text(self, text: str) -> Dict:
        endpoint = f"{self.base_url}/text/validate"
        payload = {"text": text}
        try:
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Text validation error: {str(e)}")
            return {"error": str(e)}

    def _send_audio(self, file_path: str, endpoint: str) -> Dict:
        if not os.path.exists(file_path):
            logger.error(f"Audio file not found: {file_path}")
            return {"error": "File not found"}

        try:
            with open(file_path, 'rb') as f:
                files = {"file": (os.path.basename(file_path), f, "audio/wav")}
                response = requests.post(endpoint, files=files)
                response.raise_for_status()
                return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Audio request error ({endpoint}): {str(e)}")
            return {"error": str(e)}
