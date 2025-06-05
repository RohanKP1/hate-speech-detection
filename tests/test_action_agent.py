import os
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.action_agent import ActionRecommenderAgent

@pytest.fixture
def agent():
    return ActionRecommenderAgent()

def test_recommend_action_hate_high(agent):
    classification = {'classification': 'Hate', 'confidence': 'high'}
    result = agent.recommend_action(classification)
    assert result['action'] == 'REMOVE AND BAN'
    assert result['severity'] == 'Critical'
    assert 'immediate removal' in result['reasoning']

def test_recommend_action_toxic_low(agent):
    classification = {'classification': 'Toxic', 'confidence': 'low'}
    result = agent.recommend_action(classification)
    assert result['action'] == 'FLAG FOR REVIEW'
    assert result['severity'] == 'Low'
    assert 'human judgment' in result['reasoning']

def test_recommend_action_offensive_medium(agent):
    classification = {'classification': 'Offensive', 'confidence': 'medium'}
    result = agent.recommend_action(classification)
    assert result['action'] == 'WARN USER'
    assert result['severity'] == 'Low'
    assert 'user notification' in result['reasoning']

def test_recommend_action_neutral_any_confidence(agent):
    for conf in ['high', 'medium', 'low']:
        classification = {'classification': 'Neutral', 'confidence': conf}
        result = agent.recommend_action(classification)
        assert result['action'] == 'ALLOW'
        assert result['severity'] == 'None'
        assert 'complies with community guidelines' in result['reasoning']

def test_recommend_action_ambiguous(agent):
    classification = {'classification': 'Ambiguous', 'confidence': 'medium'}
    result = agent.recommend_action(classification)
    assert result['action'] == 'FLAG FOR REVIEW'
    assert result['severity'] == 'Low'

def test_invalid_confidence_defaults_to_low(agent):
    classification = {'classification': 'Hate', 'confidence': 'unknown'}
    result = agent.recommend_action(classification)
    assert result['action'] == 'FLAG FOR REVIEW'
    assert result['severity'] == 'Medium'

def test_invalid_category_defaults_to_ambiguous(agent):
    classification = {'classification': 'Spam', 'confidence': 'high'}
    result = agent.recommend_action(classification)
    assert result['action'] == 'FLAG FOR REVIEW'
    assert result['severity'] == 'Medium'

def test_non_dict_input_returns_error(agent):
    result = agent.recommend_action("not a dict")
    assert result['action'] == 'FLAG FOR REVIEW'
    assert result['severity'] == 'Unknown'
    assert 'error' in result['reasoning'].lower()