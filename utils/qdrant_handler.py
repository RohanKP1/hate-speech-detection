from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from utils.custom_logger import CustomLogger
from core.config import Config


class QdrantHandler:
    def __init__(
        self,
        host=Config.QDRANT_HOST,
        port=Config.QDRANT_PORT,
        collection_name="policy_documents",
    ):
        self.logger = CustomLogger("QdrantHandler")
        self.collection_name = collection_name
        try:
            self.qdrant_client = QdrantClient(host=host, port=port)
            self.logger.info(f"Connected to Qdrant at {host}:{port}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Qdrant: {str(e)}")
            raise

    def ensure_collection_exists(self, vector_size=1536):
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]
            if self.collection_name not in collection_names:
                self.logger.info(f"Creating collection: {self.collection_name}")
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                self.logger.info(
                    f"Collection '{self.collection_name}' created successfully"
                )
            else:
                self.logger.debug(f"Collection '{self.collection_name}' already exists")
        except Exception as e:
            self.logger.error(f"Failed to ensure collection exists: {str(e)}")
            raise

    def delete_collection(self):
        try:
            self.qdrant_client.delete_collection(self.collection_name)
            self.logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Failed to delete collection: {str(e)}")

    def upsert_points(self, points):
        try:
            self.qdrant_client.upsert(
                collection_name=self.collection_name, points=points
            )
            self.logger.info(f"Upserted {len(points)} points to Qdrant")
        except Exception as e:
            self.logger.error(f"Failed to upsert points: {str(e)}")
            raise

    def scroll(self, limit=10000, with_payload=True, with_vectors=False):
        try:
            return self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_payload=with_payload,
                with_vectors=with_vectors,
            )
        except Exception as e:
            self.logger.error(f"Failed to scroll collection: {str(e)}")
            return ([], None)

    def get_collections(self):
        try:
            return self.qdrant_client.get_collections()
        except Exception as e:
            self.logger.error(f"Failed to get collections: {str(e)}")
            return None

    def get_collection(self):
        try:
            return self.qdrant_client.get_collection(self.collection_name)
        except Exception as e:
            self.logger.error(f"Failed to get collection info: {str(e)}")
            return None

    def search(self, query_vector, top_k=3):
        try:
            return self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
            )
        except Exception as e:
            self.logger.error(f"Failed to search collection: {str(e)}")
            return []
