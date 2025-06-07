import cv2
import pytesseract
from utils.custom_logger import CustomLogger


class ImageTranscriptionAgent:
    def __init__(self):
        self.logger = CustomLogger("ImageTranscriptionAgent")

    def transcribe_image_file(self, image_file_path: str) -> str:
        try:
            image = cv2.imread(image_file_path)
            if image is None:
                self.logger.error(f"Failed to load image: {image_file_path}")
                return ""
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            text = pytesseract.image_to_string(image_rgb)
            self.logger.info(f"Extracted text from image: {image_file_path}")
            return text
        except Exception as e:
            self.logger.error(f"Error during image transcription: {str(e)}")
            return ""
