from typing import Dict, Set, Optional
from utils.custom_logger import CustomLogger

class ActionRecommenderAgent:
    # Valid confidence levels
    VALID_CONFIDENCE_LEVELS: Set[str] = {'high', 'medium', 'low'}
    
    def __init__(self):
        self.logger = CustomLogger("ActionRecommender")
        self.action_map = {
            'Hate': {
                'high': 'REMOVE_AND_BAN',
                'medium': 'REMOVE_AND_WARN',
                'low': 'FLAG_FOR_REVIEW'
            },
            'Toxic': {
                'high': 'REMOVE_AND_WARN',
                'medium': 'REMOVE_AND_WARN',
                'low': 'FLAG_FOR_REVIEW'
            },
            'Offensive': {
                'high': 'REMOVE_AND_WARN',
                'medium': 'WARN_USER',
                'low': 'FLAG_FOR_REVIEW'
            },
            'Neutral': {
                'high': 'ALLOW',
                'medium': 'ALLOW',
                'low': 'ALLOW'
            },
            'Ambiguous': {
                'high': 'FLAG_FOR_REVIEW',
                'medium': 'FLAG_FOR_REVIEW',
                'low': 'FLAG_FOR_REVIEW'
            }
        }
        
        self.action_descriptions = {
            'REMOVE_AND_BAN': 'Remove content immediately and ban user account',
            'REMOVE_AND_WARN': 'Remove content and issue warning to user',
            'WARN_USER': 'Keep content but warn user about policy violation',
            'FLAG_FOR_REVIEW': 'Flag content for human moderator review',
            'ALLOW': 'Allow content to remain published'
        }
        self.logger.info("Initialized action recommender with predefined policies")
    
    def _validate_confidence(self, confidence: Optional[str]) -> str:
        """Validate and normalize confidence level"""
        if not confidence:
            return 'low'
        
        normalized = confidence.lower()
        if normalized not in self.VALID_CONFIDENCE_LEVELS:
            self.logger.warning(f"Invalid confidence: {confidence}, defaulting to low")
            return 'low'
        return normalized
    
    def _validate_category(self, category: Optional[str]) -> str:
        """Validate content category"""
        if not category or category not in self.action_map:
            self.logger.warning(f"Invalid category: {category}, defaulting to Ambiguous")
            return 'Ambiguous'
        return category
    
    def recommend_action(self, classification: Dict[str, str]) -> Dict[str, str]:
        """Recommend moderation action based on classification and confidence"""
        try:
            if not isinstance(classification, dict):
                raise ValueError("Classification must be a dictionary")

            # Validate inputs
            category = self._validate_category(classification.get('classification'))
            confidence = self._validate_confidence(classification.get('confidence'))
            
            self.logger.debug(f"Processing recommendation for category: {category}, confidence: {confidence}")
            
            # Get recommended action
            action = self.action_map[category][confidence]
            description = self.action_descriptions[action]
            severity = self._assess_severity(category, confidence)
            
            result = {
                'action': action,
                'description': description,
                'severity': severity,
                'reasoning': self._generate_action_reasoning(category, confidence, action)
            }
            
            self.logger.info(f"Recommended action: {action} (Severity: {severity})")
            return result
            
        except Exception as e:
            self.logger.error(f"Error recommending action: {str(e)}")
            return {
                'action': 'FLAG_FOR_REVIEW',
                'description': 'Flag for human review due to error',
                'severity': 'Unknown',
                'reasoning': f'Error in action recommendation: {str(e)}'
            }

    def _assess_severity(self, category: str, confidence: str) -> str:
        """Assess the severity level of the content"""
        try:
            if not category or not confidence:
                raise ValueError("Category and confidence are required")

            severity_matrix = {
                'Hate': {'high': 'Critical', 'medium': 'High', 'low': 'Medium'},
                'Toxic': {'high': 'High', 'medium': 'Medium', 'low': 'Low'},
                'Offensive': {'high': 'Medium', 'medium': 'Low', 'low': 'Low'},
                'Neutral': {'high': 'None', 'medium': 'None', 'low': 'None'},
                'Ambiguous': {'high': 'Medium', 'medium': 'Low', 'low': 'Low'}
            }
            
            if category not in severity_matrix:
                self.logger.warning(f"Unknown category: {category}")
                return 'Unknown'

            if confidence not in severity_matrix[category]:
                self.logger.warning(f"Unknown confidence level: {confidence}")
                return 'Unknown'

            severity = severity_matrix[category][confidence]
            self.logger.debug(f"Assessed severity for {category}/{confidence}: {severity}")
            return severity
            
        except Exception as e:
            self.logger.error(f"Error assessing severity: {str(e)}")
            return 'Unknown'
    
    def _generate_action_reasoning(self, category: str, confidence: str, action: str) -> str:
        """Generate reasoning for the recommended action"""
        try:
            base_reasoning = {
                'REMOVE_AND_BAN': f"Content classified as {category} with {confidence} confidence requires immediate removal and account suspension to prevent further violations.",
                'REMOVE_AND_WARN': f"Content classified as {category} with {confidence} confidence violates community guidelines and should be removed with user education.",
                'WARN_USER': f"Content classified as {category} with {confidence} confidence borders on policy violation and requires user notification.",
                'FLAG_FOR_REVIEW': f"Content classified as {category} with {confidence} confidence requires human judgment for proper assessment.",
                'ALLOW': f"Content classified as {category} with {confidence} confidence complies with community guidelines."
            }
            
            reasoning = base_reasoning.get(action, "Standard moderation protocols apply.")
            self.logger.debug(f"Generated reasoning for {action}: {reasoning[:50]}...")
            return reasoning
            
        except Exception as e:
            self.logger.error(f"Error generating reasoning: {str(e)}")
            return "Standard moderation protocols apply."