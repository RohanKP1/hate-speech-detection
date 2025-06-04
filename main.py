from agents.audio_agent import test, AudioTranscriptionAgent
from agents.hate_speech_agent import HateSpeechDetectionAgent
from agents.retriever_agent import HybridRetrieverAgent
from agents.reasoning_agent import PolicyReasoningAgent


if __name__ == "__main__":
    # client = AudioTranscriptionAgent()
    # txt = client.transcribe_real_time_audio()
    # print(txt)
    retrived_policy_agent = HybridRetrieverAgent("data/policy_docs")
    retrived_policy = retrived_policy_agent.retrieve_relevant_policies(
        query="hate speech",
        classification="Hate"
    )

    client = HateSpeechDetectionAgent()
    out = client.classify_text("Hello, how are you? I hate you!")
    print(out)

    explanation_agent = PolicyReasoningAgent()
    explanation = explanation_agent.generate_explanation(
        text="Hello, how are you? I hate you!",
        classification=out,
        retrieved_policies=retrived_policy
    )
    print(explanation)
