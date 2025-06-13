from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from typing import List, Dict
from core.config import Config
from utils.custom_logger import CustomLogger


class PolicyReasoningAgent:
    def __init__(self):
        self.logger = CustomLogger("PolicyReasoningAgent")
        try:
            self.model = AzureChatOpenAI(
                openai_api_version=Config.DIAL_API_VERSION,
                azure_deployment=Config.PRIMARY_MODEL_NAME,
                azure_endpoint=Config.DIAL_API_ENDPOINT,
                api_key=Config.DIAL_API_KEY,
                max_tokens=300,
                temperature=0.2,
            )
            self.logger.info(
                "Successfully initialized Azure OpenAI client with LangChain"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
            raise

    def generate_explanation(
        self, text: str, classification: Dict[str, str], retrieved_policies: List[Dict]
    ) -> str:
        """Generate detailed explanation for the classification decision"""
        try:
            # Format retrieved policies for the prompt
            policy_context = self._format_policies(retrieved_policies)

            prompt = f"""
            Based on the content moderation analysis and relevant policies, provide a clear explanation for why the content was classified as "{classification['classification']}".

            Original Text: "{text}"
            Classification: {classification['classification']}
            Initial Reason: {classification['reason']}
            Confidence: {classification['confidence']}

            Relevant Policies:
            {policy_context}

            Provide a comprehensive explanation that:
            1. References specific policies that apply
            2. Explains which aspects of the content violate or comply with guidelines
            3. Justifies the confidence level
            4. Considers any contextual factors

            Keep the explanation clear, factual, and professional.
            """

            self.logger.debug(f"Sending explanation request for text: {text[:50]}...")

            messages = [
                SystemMessage(
                    content="You are an expert policy analyst specializing in content moderation and hate speech detection."
                ),
                HumanMessage(content=prompt),
            ]

            response = self.model.invoke(messages)

            result = response.content
            self.logger.info("Successfully generated explanation")
            return result.strip() if result else ""

        except Exception as e:
            error_msg = f"Unable to generate detailed explanation: {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    def _format_policies(self, policies: List[Dict]) -> str:
        """Format retrieved policies for inclusion in prompt"""
        if not policies:
            return "No specific policies retrieved."

        formatted = ""
        for i, policy in enumerate(policies, 1):
            formatted += (
                f"\n{i}. {policy['source']} (Relevance: {policy['relevance_score']}):\n"
            )
            formatted += f"   {policy['content']}\n"

        return formatted
