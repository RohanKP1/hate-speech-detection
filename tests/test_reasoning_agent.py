import os
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.reasoning_agent import PolicyReasoningAgent

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
    agent = PolicyReasoningAgent()
    return agent

def test_generate_explanation_success(monkeypatch, agent):
    dummy_explanation = "This content was classified as Hate because it contains explicit hate speech."
    agent.model = DummyModel(dummy_explanation)
    text = "Some hateful text"
    classification = {
        "classification": "Hate",
        "reason": "Contains hate speech",
        "confidence": "high"
    }
    retrieved_policies = [
        {"source": "Policy A", "relevance_score": 98.5, "content": "No hate speech allowed."}
    ]
    result = agent.generate_explanation(text, classification, retrieved_policies)
    assert dummy_explanation in result

def test_generate_explanation_empty(monkeypatch, agent):
    agent.model = DummyModel("")
    text = "Neutral text"
    classification = {
        "classification": "Neutral",
        "reason": "No violation",
        "confidence": "high"
    }
    retrieved_policies = []
    result = agent.generate_explanation(text, classification, retrieved_policies)
    assert result == ""

def test_generate_explanation_exception(monkeypatch, agent):
    class ErrorModel:
        def invoke(self, messages):
            raise Exception("Model error")
    agent.model = ErrorModel()
    text = "Some text"
    classification = {
        "classification": "Offensive",
        "reason": "Offensive language",
        "confidence": "medium"
    }
    retrieved_policies = []
    result = agent.generate_explanation(text, classification, retrieved_policies)
    assert "Unable to generate detailed explanation" in result

def test_format_policies_none(agent):
    formatted = agent._format_policies([])
    assert "No specific policies" in formatted

def test_format_policies_multiple(agent):
    policies = [
        {"source": "Policy X", "relevance_score": 99.1, "content": "Policy X content."},
        {"source": "Policy Y", "relevance_score": 87.5, "content": "Policy Y content."}
    ]
    formatted = agent._format_policies(policies)
    assert "1. Policy X" in formatted
    assert "2. Policy Y" in formatted