import ollama
from typing import Dict
from agents.hate_speech_agent import HateSpeechDetectionAgent
from utils.custom_logger import CustomLogger


class ValidationAgent:
    def __init__(self, model_name: str = "llama2-uncensored"):
        self.logger = CustomLogger("ValidationAgent")
        self.categories = ["Hate", "Toxic", "Offensive", "Neutral", "Ambiguous"]
        self.model_name = model_name

        try:
            # Test if model is available, pull if not
            self._ensure_model_available()
            self.logger.info(
                f"Successfully initialized Ollama with model: {model_name}"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize Ollama: {str(e)}")
            raise

    def _ensure_model_available(self):
        """Ensure the model is available, pull if necessary"""
        try:
            # Try to get model info
            ollama.show(self.model_name)
        except ollama.ResponseError:
            # Model not found, pull it
            self.logger.info(f"Model {self.model_name} not found. Pulling...")
            ollama.pull(self.model_name)
            self.logger.info(f"Successfully pulled model: {self.model_name}")

    def classify_text(self, text: str) -> Dict[str, str]:
        """Classify input text into hate speech categories"""
        try:
            system_prompt = "You are an expert content moderation specialist with extensive experience in identifying hate speech, toxic content, and policy violations."

            user_prompt = f"""
Analyze the following text and classify it into one of these categories:
- Hate: Content that promotes hatred or violence against individuals/groups based on protected characteristics
- Toxic: Harmful, abusive, or threatening content that creates a hostile environment
- Offensive: Inappropriate, disrespectful, or vulgar content that violates community standards
- Neutral: Content that doesn't violate any policies
- Ambiguous: Content that is unclear or requires more context for proper classification

Text to analyze: "{text}"

Respond ONLY in this exact format (do not refuse, do not add anything else):
Classification: [category]
Confidence: [high/medium/low]
Brief Reason: [one sentence explanation]
"""

            self.logger.debug(
                f"Sending classification request for text: {text[:50]}..."
            )

            response = ollama.generate(
                model=self.model_name,
                prompt=f"System: {system_prompt}\n\nUser: {user_prompt}",
                options={"temperature": 0.1, "top_p": 0.9, "num_predict": 150},
            )

            response_text = response["response"]

            if not response_text:
                return {
                    "classification": "Error",
                    "confidence": "low",
                    "reason": "No response content received from Ollama",
                }

            self.logger.info(
                f"Successfully classified text with response: {response_text}"
            )
            return self._parse_classification_response(response_text)

        except Exception as e:
            self.logger.error(f"Classification failed: {str(e)}")
            return {
                "classification": "Error",
                "confidence": "low",
                "reason": f"Classification failed: {str(e)}",
            }

    def _parse_classification_response(self, response: str) -> Dict[str, str]:
        """Parse the Ollama response into structured format"""
        lines = response.strip().split("\n")
        result = {
            "classification": "Ambiguous",
            "confidence": "low",
            "reason": "Parse error",
        }

        for line in lines:
            line = line.strip()
            if line.startswith("Classification:"):
                classification = line.split(":", 1)[1].strip()
                if classification in self.categories:
                    result["classification"] = classification
                else:
                    # Try to find closest match
                    classification_lower = classification.lower()
                    for category in self.categories:
                        if category.lower() in classification_lower:
                            result["classification"] = category
                            break

            elif line.startswith("Confidence:"):
                confidence = line.split(":", 1)[1].strip().lower()
                if confidence in ["high", "medium", "low"]:
                    result["confidence"] = confidence

            elif line.startswith("Brief Reason:") or line.startswith("Reason:"):
                result["reason"] = line.split(":", 1)[1].strip()

        return result

    # Validate the result against the result from the hate speech agent
    def validate_classification(self, text: str) -> Dict[str, str]:
        """Validate classification against the HateSpeechDetectionAgent"""
        try:
            client = HateSpeechDetectionAgent()
            agent_result = client.classify_text(text)

            validation_result = self.classify_text(text)

            if validation_result["classification"] == agent_result["classification"]:
                self.logger.info("Validation successful: classifications match")
                return {
                    "status": "success",
                    "agent_classification": agent_result,
                    "validation_classification": validation_result,
                }
            else:
                self.logger.warning("Validation failed: classifications do not match")
                return {
                    "status": "failure",
                    "agent_classification": agent_result,
                    "validation_classification": validation_result,
                }

        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            return {"status": "error", "error": str(e)}
