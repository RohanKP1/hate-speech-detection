from openai import OpenAI
from core.config import Config
from utils.custom_logger import CustomLogger
import pyaudio
import wave
import tempfile
import os


class AudioTranscriptionAgent:
    def __init__(self):
        self.logger = CustomLogger("AudioTranscriptionAgent")
        try:
            self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
            self.logger.info("Successfully initialized OpenAI client")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise

    def transcribe_audio_file(self, audio_file_path: str) -> str:
        """Transcribe audio file using OpenAI's transcription model"""
        try:
            with open(audio_file_path, "rb") as audio_file:
                self.logger.debug(
                    f"Sending transcription request for file: {audio_file_path}"
                )
                transcription = self.client.audio.transcriptions.create(
                    model=Config.AUDIO_MODEL_NAME,
                    file=audio_file,
                    response_format="text",
                    language="en",
                )
                self.logger.info("Audio transcription completed successfully")
                return transcription
        except Exception as e:
            self.logger.error(f"Failed to transcribe audio: {str(e)}")
            raise

    # Create transcribe_real_time_audio method using pyaudio and OpenAI's real-time transcription capabilities
    def transcribe_real_time_audio(self):
        """Transcribe real-time audio using OpenAI's real-time transcription capabilities"""
        # Audio Recording Configuration
        CHUNK = 8192  # Larger buffer for better processing efficiency
        FORMAT = pyaudio.paFloat32  # 32-bit float for studio-quality audio
        CHANNELS = 1  # Mono channel - better for speech recognition
        RATE = 44100  # CD-quality sampling rate
        RECORD_SECONDS = 2  # 5 seconds for better context

        # Advanced configurations
        THRESHOLD = 0.03  # Sound detection threshold
        SILENCE_LIMIT = 1  # Seconds of silence before stopping

        # For potential future enhancements:
        # - Voice activity detection (VAD)
        # - Automatic gain control
        # - Real-time noise suppression

        try:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
            )

            self.logger.info("* Recording started")
            frames = []

            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)

            self.logger.info("* Recording finished")

            stream.stop_stream()
            stream.close()
            p.terminate()

            # Save the recorded data as a WAV file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                wf = wave.open(temp_audio.name, "wb")
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b"".join(frames))
                wf.close()

                # Transcribe the temporary audio file
                transcription = self.transcribe_audio_file(temp_audio.name)

            # Clean up temporary file
            os.unlink(temp_audio.name)
            return transcription

        except Exception as e:
            self.logger.error(
                f"Failed to capture or transcribe real-time audio: {str(e)}"
            )
            raise


def test():
    agent = AudioTranscriptionAgent()
    transcription = agent.transcribe_audio_file("assets/audio/apple_juice.flac")
    print(transcription)
