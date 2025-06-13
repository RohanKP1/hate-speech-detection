from typing import Dict, Any
import traceback
from utils.custom_logger import CustomLogger


class ErrorHandlerAgent:
    def __init__(self):
        self.logger = CustomLogger("ErrorHandler")

    def handle_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """Handle errors gracefully and return user-friendly message"""
        error_type = type(error).__name__
        error_message = str(error)

        # Log the error for debugging
        self.logger.error(f"Error in {context}: {error_type} - {error_message}")
        self.logger.error(f"Traceback: {traceback.format_exc()}")

        # Return user-friendly error response
        if "API" in error_message or "openai" in error_message.lower():
            return {
                "error": True,
                "type": "API Error",
                "message": "There was an issue connecting to the AI service. Please try again later.",
                "suggestion": "Check your internet connection and API credentials.",
            }
        elif "file" in error_message.lower() or "directory" in error_message.lower():
            return {
                "error": True,
                "type": "File Error",
                "message": "There was an issue accessing policy documents.",
                "suggestion": "Ensure all policy files are present in the data/policy_docs/ directory.",
            }
        elif "embedding" in error_message.lower() or "faiss" in error_message.lower():
            return {
                "error": True,
                "type": "Embedding Error",
                "message": "There was an issue with the document search system.",
                "suggestion": "The system may need to be restarted to rebuild the search index.",
            }
        else:
            return {
                "error": True,
                "type": "System Error",
                "message": "An unexpected error occurred. Please try again.",
                "suggestion": "If the problem persists, contact the system administrator.",
            }

    def validate_input(self, text: str) -> Dict[str, Any]:
        """Validate user input"""
        if not text or not text.strip():
            self.logger.warning("Empty input received")
            return {
                "valid": False,
                "error": "Empty input",
                "message": "Please enter some text to analyze.",
            }

        if len(text.strip()) < 3:
            self.logger.warning("Input too short")
            return {
                "valid": False,
                "error": "Input too short",
                "message": "Please enter at least 3 characters for analysis.",
            }

        if len(text) > 5000:
            self.logger.warning("Input exceeds maximum length")
            return {
                "valid": False,
                "error": "Input too long",
                "message": "Please limit input to 5000 characters or less.",
            }

        self.logger.debug("Input validation successful")
        return {"valid": True}
