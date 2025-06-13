import os
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agents.hate_speech_agent import HateSpeechDetectionAgent


class DummyModel:
    """A dummy model to mock the LangChain AzureChatOpenAI model."""

    def __init__(self, response_content):
        self.response_content = response_content

    def invoke(self, messages):
        class DummyResponse:
            def __init__(self, content):
                self.content = content

        return DummyResponse(self.response_content)


@pytest.fixture
def agent(monkeypatch):
    agent = HateSpeechDetectionAgent()
    return agent


def test_classify_text_hate(monkeypatch, agent):
    # Mock the model's invoke method
    response = (
        "Classification: Hate\n"
        "Confidence: high\n"
        "Brief Reason: Contains explicit hate speech."
    )
    agent.model = DummyModel(response)
    result = agent.classify_text("Some hateful text")
    assert result["classification"] == "Hate"
    assert result["confidence"] == "high"
    assert "hate speech" in result["reason"].lower()


def test_classify_text_neutral(monkeypatch, agent):
    response = (
        "Classification: Neutral\n"
        "Confidence: high\n"
        "Brief Reason: No policy violation detected."
    )
    agent.model = DummyModel(response)
    result = agent.classify_text("Hello, how are you?")
    assert result["classification"] == "Neutral"
    assert result["confidence"] == "high"
    assert "no policy violation" in result["reason"].lower()


def test_classify_text_error(monkeypatch, agent):
    # Simulate model returning None
    class ErrorModel:
        def invoke(self, messages):
            class DummyResponse:
                content = None

            return DummyResponse()

    agent.model = ErrorModel()
    result = agent.classify_text("Test")
    assert result["classification"] == "Error"
    assert result["confidence"] == "Low"
    assert "no response content" in result["reason"].lower()


def test_parse_classification_response_partial(agent):
    # Only classification present
    response = "Classification: Offensive"
    parsed = agent._parse_classification_response(response)
    assert parsed["classification"] == "Offensive"
    assert parsed["confidence"] == "Low"  # default
    assert parsed["reason"] == "Parse error"  # default


def test_parse_classification_response_full(agent):
    response = (
        "Classification: Toxic\n"
        "Confidence: medium\n"
        "Brief Reason: Contains abusive language."
    )
    parsed = agent._parse_classification_response(response)
    assert parsed["classification"] == "Toxic"
    assert parsed["confidence"] == "medium"
    assert parsed["reason"] == "Contains abusive language."
