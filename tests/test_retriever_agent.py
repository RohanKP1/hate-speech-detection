import os
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.retriever_agent import HybridRetrieverAgent
from utils.custom_logger import CustomLogger

class DummyEmbeddings:
    def __init__(self, docs=None, search_results=None):
        self.docs = docs or []
        self._search_results = search_results or []

    def load_documents(self, policy_docs_path):
        return self.docs

    def create_embeddings(self):
        return True

    def search(self, query, top_k):
        # Return dummy search results
        return self._search_results[:top_k]

@pytest.fixture
def dummy_agent(monkeypatch):
    # Patch PolicyEmbeddings in HybridRetrieverAgent to use DummyEmbeddings
    dummy_docs = [
        {"content": "Policy 1 content", "source": "policy1.txt", "chunk_id": 0},
        {"content": "Policy 2 content", "source": "policy2.txt", "chunk_id": 0},
    ]
    dummy_search_results = [
        {"content": "Policy 1 content", "source": "policy1.txt", "chunk_id": 0, "score": 0.95, "rank": 1},
        {"content": "Policy 2 content", "source": "policy2.txt", "chunk_id": 0, "score": 0.85, "rank": 2},
    ]
    def dummy_init(self, policy_docs_path):
        self.logger = CustomLogger("dummy_logger")
        self.embeddings = DummyEmbeddings(dummy_docs, dummy_search_results)
        self.policy_docs_path = policy_docs_path

    monkeypatch.setattr("agents.retriever_agent.HybridRetrieverAgent.__init__", dummy_init)
    return HybridRetrieverAgent("dummy_path")

def test_retrieve_relevant_policies_returns_formatted_results(dummy_agent):
    query = "What is the hate speech policy?"
    classification = "Hate"
    results = dummy_agent.retrieve_relevant_policies(query, classification, top_k=2)
    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0]['content'] == "Policy 1 content"
    assert results[0]['source'] == "Policy1"
    assert results[0]['relevance_score'] == 0.95
    assert results[0]['rank'] == 1

def test_retrieve_relevant_policies_empty_results(monkeypatch):
    def dummy_init(self, policy_docs_path):
        self.logger = CustomLogger("dummy_logger")
        self.embeddings = DummyEmbeddings([], [])
        self.policy_docs_path = policy_docs_path
    monkeypatch.setattr("agents.retriever_agent.HybridRetrieverAgent.__init__", dummy_init)
    agent = HybridRetrieverAgent("dummy_path")
    results = agent.retrieve_relevant_policies("query", "Neutral", top_k=2)
    assert results == []

def test_retrieve_relevant_policies_handles_exception(monkeypatch):
    class FailingEmbeddings(DummyEmbeddings):
        def search(self, query, top_k):
            raise Exception("Search failed")
    def dummy_init(self, policy_docs_path):
        self.logger = CustomLogger("dummy_logger")
        self.embeddings = FailingEmbeddings()
        self.policy_docs_path = policy_docs_path
    monkeypatch.setattr("agents.retriever_agent.HybridRetrieverAgent.__init__", dummy_init)
    agent = HybridRetrieverAgent("dummy_path")
    results = agent.retrieve_relevant_policies("query", "Hate", top_k=1)
    assert results == []