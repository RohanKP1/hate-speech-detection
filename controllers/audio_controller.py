from controllers.text_controller import TextController
from agents.audio_agent import AudioTranscriptionAgent
from utils.custom_logger import CustomLogger


class AudioController:
    def __init__(self, policy_docs=None):
        self.logger = CustomLogger("AudioController")
        self.audio_agent = AudioTranscriptionAgent()
        self.text_controller = TextController(policy_docs)
        self.logger.info("AudioController initialized with all agents")

    def transcribe_audio(self, audio_file: str):
        """Transcribe audio file and return text."""
        try:
            self.logger.info(f"Transcribing audio file: {audio_file}")
            transcription = self.audio_agent.transcribe_audio_file(audio_file)
            self.logger.info(f"Transcription result: {transcription}")
            return transcription
        except Exception as e:
            error_message = f"An error occurred during audio transcription: {str(e)}"
            self.logger.error(error_message)
            return {"error": error_message}

    def analyze_audio(self, audio_file: str):
        """Analyze audio file for hate speech."""
        transcription = self.transcribe_audio(audio_file)
        if isinstance(transcription, dict):
            return transcription
        analysis_result = self.text_controller.analyze_text(transcription)
        # Add your analysis logic here
        return {
            "hate_speech": analysis_result.get("hate_speech"),
            "policies": analysis_result.get("policies"),
            "reasoning": analysis_result.get("reasoning"),
            "action": analysis_result.get("action"),
            "transcription": transcription,  # Ensure this is present
        }
