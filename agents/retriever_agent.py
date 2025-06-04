from utils.embeddings import PolicyEmbeddings
from utils.custom_logger import CustomLogger

class HybridRetrieverAgent:
    def __init__(self, policy_docs_path: str):
        self.logger = CustomLogger("HybridRetriever")
        self.embeddings = PolicyEmbeddings()
        self.policy_docs_path = policy_docs_path
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize embeddings for policy documents"""
        try:
            # Load documents
            documents = self.embeddings.load_documents(self.policy_docs_path)
            
            # Create embeddings
            self.embeddings.create_embeddings()
            
            self.logger.info(f"Loaded {len(documents)} document chunks from {self.policy_docs_path}")
            
        except Exception as e:
            self.logger.error(f"Error initializing embeddings: {str(e)}")
            raise
    
    def retrieve_relevant_policies(self, query: str, classification: str, top_k: int = 5) -> list:
        """Retrieve relevant policy documents based on query and classification"""
        try:
            # Enhance query with classification for better retrieval
            enhanced_query = f"{query} {classification} hate speech moderation policy"
            self.logger.debug(f"Enhanced query: {enhanced_query[:100]}...")
            
            # Search for relevant documents
            results = self.embeddings.search(enhanced_query, top_k)
            
            # Format results for easy consumption
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'content': result['content'],
                    'source': result['source'].replace('.txt', '').replace('_', ' ').title(),
                    'relevance_score': round(result['score'], 3),
                    'rank': result['rank']
                })
            
            self.logger.info(f"Retrieved {len(formatted_results)} relevant policies")
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error retrieving policies: {str(e)}")
            return []