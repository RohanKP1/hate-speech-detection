from agents.audio_agent import test, AudioTranscriptionAgent
from agents.hate_speech_agent import HateSpeechDetectionAgent
from agents.retriever_agent import HybridRetrieverAgent
from agents.reasoning_agent import PolicyReasoningAgent
from agents.validation_agent import ValidationAgent

if __name__ == "__main__":
    # client = AudioTranscriptionAgent()
    # txt = client.transcribe_real_time_audio()
    # print(txt)
    # retrived_policy_agent = HybridRetrieverAgent("data/policy_docs")
    # retrived_policy = retrived_policy_agent.retrieve_relevant_policies(
    #     query="hate speech",
    #     classification="Hate"
    # )

    # client = HateSpeechDetectionAgent()
    # out = client.classify_text("Hello, how are you? I hate you!")
    # print(out)

    client2 = ValidationAgent()
    # out2 = client2.classify_text("Hello, how are you? I hate you!")
    # print(out2)

    # Validate the result against the result from the hate speech agent
    validation_result = client2.validate_classification("Hello, how are you? I hate you!")
    print(validation_result)

    # explanation_agent = PolicyReasoningAgent()
    # explanation = explanation_agent.generate_explanation(
    #     text="Hello, how are you? I hate you!",
    #     classification=out,
    #     retrieved_policies=retrived_policy
    # )
    # print(explanation)
