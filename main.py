from agents.audio_agent import test, AudioTranscriptionAgent


if __name__ == "__main__":
    client = AudioTranscriptionAgent()
    txt = client.transcribe_real_time_audio()
    print(txt)
