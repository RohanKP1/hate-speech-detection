from agents.hate_speech_agent import HateSpeechDetectionAgent
from agents.reasoning_agent import PolicyReasoningAgent
from agents.retriever_agent import HybridRetrieverAgent
from agents.error_handler import ErrorHandlerAgent
from agents.action_agent import ActionRecommenderAgent
from agents.validation_agent import ValidationAgent
from utils.custom_logger import CustomLogger


class TextController:
    def __init__(self, policy_docs):
        self.logger = CustomLogger("TextController")
        self.hate_speech_agent = HateSpeechDetectionAgent()
        self.reasoning_agent = PolicyReasoningAgent()
        self.retriever_agent = HybridRetrieverAgent(policy_docs)
        self.error_handler = ErrorHandlerAgent()
        self.action_agent = ActionRecommenderAgent()
        self.validation_agent = ValidationAgent()
        self.logger.info("TextController initialized with all agents")

    def analyze_text(self, text: str):
        """Analyze input text and return all results in a structured dict."""
        try:
            self.logger.info(f"Processing text: {text}")

            # Step 1: Detect hate speech
            hate_speech_result = self.hate_speech_agent.classify_text(text)
            self.logger.info(f"Hate speech detection result: {hate_speech_result}")

            # Extract classification and confidence for downstream steps
            classification = hate_speech_result.get("classification", "Ambiguous")
            confidence = hate_speech_result.get("confidence", "low")

            # Step 2: Retrieve relevant policies
            policies = self.retriever_agent.retrieve_relevant_policies(
                text, classification, top_k=5
            )
            self.logger.info(f"Retrieved policies: {policies}")

            if not policies:
                error_message = "No relevant policies found"
                self.error_handler.handle_error(RuntimeError(error_message))
                return {"error": error_message}

            # Step 3: Reason about policy
            reasoning_result = self.reasoning_agent.generate_explanation(
                text, hate_speech_result, policies
            )
            self.logger.info(f"Policy reasoning result: {reasoning_result}")

            # Step 4: Recommend action
            action = self.action_agent.recommend_action(
                {"classification": classification, "confidence": confidence}
            )
            self.logger.info(f"Recommended action: {action}")

            # Step 5: Return all results
            return {
                "hate_speech": hate_speech_result,
                "policies": policies,
                "reasoning": reasoning_result,
                "action": action,
            }

        except Exception as e:
            error_message = f"An error occurred while processing text: {str(e)}"
            self.error_handler.handle_error(RuntimeError(error_message))
            return {"error": error_message}

    def validate_classification(self, text: str):
        """Validate input text for processing."""
        try:
            self.logger.info(f"Validating text: {text}")
            validation_result = self.validation_agent.validate_classification(text)
            self.logger.info(f"Validation result: {validation_result}")
            return validation_result
        except Exception as e:
            error_message = f"An error occurred during validation: {str(e)}"
            self.error_handler.handle_error(RuntimeError(error_message))
            return {"error": error_message}
