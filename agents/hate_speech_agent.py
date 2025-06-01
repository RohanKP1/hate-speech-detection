from openai import AzureOpenAI
from typing import Dict
from core.config import Config
from utils.custom_logger import CustomLogger

class HateSpeechDetectionAgent:
    def __init__(self):
        self.logger = CustomLogger("HateSpeechAgent")
        self.categories = ['Hate', 'Toxic', 'Offensive', 'Neutral', 'Ambiguous']
        
        try:
            self.client = AzureOpenAI(
                api_version=Config.DIAL_API_VERSION,
                azure_endpoint=Config.DIAL_API_ENDPOINT,
                api_key=Config.DIAL_API_KEY
            )
            self.logger.info("Successfully initialized Azure OpenAI client")
        except Exception as e:
            self.logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
            raise
    
    def classify_text(self, text: str) -> Dict[str, str]:
        """Classify input text into hate speech categories"""
        try:
            prompt = f"""
            Analyze the following text and classify it into one of these categories:
            - Hate: Content that promotes hatred or violence against individuals/groups based on protected characteristics
            - Toxic: Harmful, abusive, or threatening content that creates a hostile environment
            - Offensive: Inappropriate, disrespectful, or vulgar content that violates community standards
            - Neutral: Content that doesn't violate any policies
            - Ambiguous: Content that is unclear or requires more context for proper classification
            
            Text to analyze: "{text}"
            
            Respond in this exact format:
            Classification: [category]
            Confidence: [high/medium/low]
            Brief Reason: [one sentence explanation]
            """
            
            self.logger.debug(f"Sending classification request for text: {text[:50]}...")
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert content moderation specialist."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.1
            )
            
            result = response.choices[0].message.content
            if result is None:
                return {
                    'classification': 'Error',
                    'confidence': 'low',
                    'reason': 'No response content received'
                }
            self.logger.info(f"Successfully classified text with response: {result}")
            return self._parse_classification_response(result)
            
        except Exception as e:
            self.logger.error(f"Classification failed: {str(e)}")
            return {
                'classification': 'Error',
                'confidence': 'low',
                'reason': f'Classification failed: {str(e)}'
            }
    
    def _parse_classification_response(self, response: str) -> Dict[str, str]:
        """Parse the OpenAI response into structured format"""
        lines = response.strip().split('\n')
        result = {'classification': 'Ambiguous', 'confidence': 'low', 'reason': 'Parse error'}
        
        for line in lines:
            if line.startswith('Classification:'):
                result['classification'] = line.split(':', 1)[1].strip()
            elif line.startswith('Confidence:'):
                result['confidence'] = line.split(':', 1)[1].strip()
            elif line.startswith('Brief Reason:'):
                result['reason'] = line.split(':', 1)[1].strip()
        
        return result