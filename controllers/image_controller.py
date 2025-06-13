from agents.image_agent import ImageTranscriptionAgent
from controllers.text_controller import TextController
from utils.custom_logger import CustomLogger


class ImageController:
    def __init__(self, policy_docs=None):
        self.logger = CustomLogger("ImageController")
        self.image_agent = ImageTranscriptionAgent()
        self.text_controller = TextController(policy_docs)
        self.logger.info("ImageController initialized with all agents")

    def analyze_image(self, image_file: str):
        try:
            transcription = self.image_agent.transcribe_image_file(image_file)
            if not transcription:
                return {"error": "No text extracted from image."}
            analysis = self.text_controller.analyze_text(transcription)
            return {
                "hate_speech": analysis.get("hate_speech", {}),
                "policies": analysis.get("policies", {}),
                "reasoning": analysis.get("reasoning", ""),
                "action": analysis.get("action", {}),
                "transcription": transcription,
            }
        except Exception as e:
            self.logger.error(f"Error analyzing image: {str(e)}")
            return {"error": str(e)}
