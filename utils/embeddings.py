from langchain_openai import AzureOpenAIEmbeddings
import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
from qdrant_client.models import PointStruct
from core.config import Config
from utils.custom_logger import CustomLogger
from utils.qdrant_handler import QdrantHandler
from core.config import Config


class PolicyEmbeddings:
    def __init__(
        self,
        qdrant_host=Config.QDRANT_HOST,
        qdrant_port=Config.QDRANT_PORT,
        collection_name="policy_documents",
        storage_path="logs/policy_embeddings",
    ):
        self.logger = CustomLogger("PolicyEmbeddings")
        self.storage_path = storage_path
        self.collection_name = collection_name
        self.documents = []

        os.makedirs(storage_path, exist_ok=True)
        self.metadata_path = os.path.join(storage_path, "metadata.json")

        # Qdrant handler
        self.qdrant_handler = QdrantHandler(
            host=qdrant_host, port=qdrant_port, collection_name=collection_name
        )

        # OpenAI embeddings client
        try:
            self.client = AzureOpenAIEmbeddings(
                openai_api_version=Config.DIAL_API_VERSION,
                azure_deployment=Config.EMBEDDING_MODEL_NAME,
                azure_endpoint=Config.DIAL_API_ENDPOINT,
                api_key=Config.DIAL_API_KEY,
                check_embedding_ctx_length=False,
            )
            self.logger.info("Successfully initialized OpenAI client")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise

    def _calculate_content_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _calculate_directory_hash(self, docs_path: str) -> str:
        file_hashes = []
        if os.path.exists(docs_path):
            for filename in sorted(os.listdir(docs_path)):
                if filename.endswith(".txt"):
                    filepath = os.path.join(docs_path, filename)
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                        file_hashes.append(
                            f"{filename}:{self._calculate_content_hash(content)}"
                        )
        combined_hash = hashlib.sha256(
            "|".join(file_hashes).encode("utf-8")
        ).hexdigest()
        return combined_hash

    def _load_metadata(self) -> Dict[str, Any]:
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load metadata: {str(e)}")
        return {}

    def _save_metadata(self, metadata: Dict[str, Any]):
        try:
            with open(self.metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {str(e)}")
            raise

    def _has_policy_changed(self, docs_path: str) -> tuple[bool, str]:
        current_hash = self._calculate_directory_hash(docs_path)
        metadata = self._load_metadata()
        stored_hash = metadata.get("policy_versions", {}).get(docs_path)
        if stored_hash is None:
            return True, current_hash
        return stored_hash != current_hash, current_hash

    def _update_policy_version(self, docs_path: str, version_hash: str):
        try:
            metadata = self._load_metadata()
            if "policy_versions" not in metadata:
                metadata["policy_versions"] = {}
            metadata["policy_versions"][docs_path] = version_hash
            metadata["last_updated"] = datetime.now().isoformat()
            self._save_metadata(metadata)
        except Exception as e:
            self.logger.error(f"Error updating policy version: {str(e)}")
            raise

    def _load_documents_from_qdrant(self) -> List[Dict[str, Any]]:
        try:
            response = self.qdrant_handler.scroll(
                limit=10000, with_payload=True, with_vectors=False
            )
            documents = []
            for point in response[0]:
                payload = point.payload
                documents.append(
                    {
                        "content": payload["content"],
                        "source": payload["source"],
                        "chunk_id": payload["chunk_id"],
                        "content_hash": payload["content_hash"],
                        "doc_id": payload["doc_id"],
                        "qdrant_id": point.id,
                    }
                )
            self.logger.info(f"Loaded {len(documents)} documents from Qdrant")
            return documents
        except Exception as e:
            self.logger.error(f"Failed to load documents from Qdrant: {str(e)}")
            return []

    def load_documents(self, docs_path: str) -> List[Dict[str, Any]]:
        self.logger.info(f"Loading documents from {docs_path}")
        self.qdrant_handler.ensure_collection_exists(vector_size=1536)
        has_changed, current_hash = self._has_policy_changed(docs_path)

        if not has_changed:
            self.logger.info("No policy changes detected, loading existing documents")
            self.documents = self._load_documents_from_qdrant()
            if self.documents:
                return self.documents
            else:
                self.logger.warning("No documents found in Qdrant, will process files")
                has_changed = True

        if has_changed:
            self.logger.info(
                "Policy changes detected or no existing data, processing documents"
            )
            try:
                self.qdrant_handler.delete_collection()
                self.logger.info(f"Cleared existing collection: {self.collection_name}")
            except:
                pass
            self.qdrant_handler.ensure_collection_exists(vector_size=1536)
            documents = []
            doc_counter = 0
            for filename in os.listdir(docs_path):
                if filename.endswith(".txt"):
                    filepath = os.path.join(docs_path, filename)
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        content_hash = self._calculate_content_hash(content)
                        documents.append(
                            {
                                "content": content,
                                "source": filename,
                                "chunk_id": 0,
                                "content_hash": content_hash,
                                "doc_id": doc_counter,
                            }
                        )
                        doc_counter += 1
            self.documents = documents
            self.logger.info(f"Loaded {len(documents)} policy documents (no chunking)")
            self._update_policy_version(docs_path, current_hash)
        return self.documents

    def create_embeddings(self):
        try:
            if not self.documents:
                raise ValueError("No documents loaded. Call load_documents() first.")
            existing_docs = self._load_documents_from_qdrant()
            if len(existing_docs) == len(self.documents):
                self.logger.info("Embeddings already exist in Qdrant")
                return
            self.logger.info("Creating embeddings for documents")
            texts = [doc["content"] for doc in self.documents]
            batch_size = 100
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                self.logger.debug(
                    f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}"
                )
                response = self.client.embed_documents(batch)
                all_embeddings.extend(response)
            points = []
            for i, (doc, embedding) in enumerate(zip(self.documents, all_embeddings)):
                point = PointStruct(
                    id=i,
                    vector=embedding,
                    payload={
                        "content": doc["content"],
                        "source": doc["source"],
                        "chunk_id": doc["chunk_id"],
                        "content_hash": doc["content_hash"],
                        "doc_id": doc["doc_id"],
                    },
                )
                points.append(point)
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                self.qdrant_handler.upsert_points(batch)
                self.logger.debug(f"Uploaded batch {i//batch_size + 1} to Qdrant")
            self.logger.info(f"Created and stored {len(points)} embeddings in Qdrant")
        except Exception as e:
            self.logger.error(f"Failed to create embeddings: {str(e)}")
            raise

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        try:
            self.logger.debug(f"Searching for query: {query[:50]}...")
            query_embedding = self.client.embed_query(query)
            search_results = self.qdrant_handler.search(
                query_vector=query_embedding, top_k=top_k
            )
            results = []
            for i, result in enumerate(search_results):
                results.append(
                    {
                        "content": result.payload["content"],
                        "source": result.payload["source"],
                        "chunk_id": result.payload["chunk_id"],
                        "score": round(float(result.score) * 100, 2),
                        "rank": i + 1,
                    }
                )
            self.logger.info(f"Found {len(results)} matching documents")
            return results
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            raise

    def force_refresh(self, docs_path: str):
        self.logger.info("Forcing refresh of embeddings")
        try:
            try:
                self.qdrant_handler.delete_collection()
                self.logger.info(f"Deleted collection: {self.collection_name}")
            except:
                pass
            metadata = self._load_metadata()
            if (
                "policy_versions" in metadata
                and docs_path in metadata["policy_versions"]
            ):
                del metadata["policy_versions"][docs_path]
                self._save_metadata(metadata)
            self.documents = []
            self.load_documents(docs_path)
            self.create_embeddings()
        except Exception as e:
            self.logger.error(f"Error during force refresh: {str(e)}")
            raise

    def get_storage_stats(self) -> Optional[Dict[str, Any]]:
        try:
            metadata = self._load_metadata()
            try:
                collection_info = self.qdrant_handler.get_collection()
                vector_count = collection_info.points_count
                vector_size = collection_info.config.params.vectors.size
            except:
                vector_count = 0
                vector_size = 0
            stats = {
                "document_count": len(self.documents) if self.documents else 0,
                "embedding_dimension": vector_size,
                "vector_count": vector_count,
                "collection_name": self.collection_name,
                "storage_path": self.storage_path,
                "policy_versions": metadata.get("policy_versions", {}),
                "last_updated": metadata.get("last_updated"),
            }
            if os.path.exists(self.metadata_path):
                stats["metadata_size_mb"] = round(
                    os.path.getsize(self.metadata_path) / (1024 * 1024), 2
                )
            else:
                stats["metadata_size_mb"] = 0
            return stats
        except Exception as e:
            self.logger.error(f"Error getting storage stats: {str(e)}")
            return None

    def health_check(self) -> Dict[str, Any]:
        try:
            collections = self.qdrant_handler.get_collections()
            qdrant_healthy = True
            collection_names = [c.name for c in collections.collections]
            collection_exists = self.collection_name in collection_names
            collection_stats = None
            if collection_exists:
                collection_info = self.qdrant_handler.get_collection()
                collection_stats = {
                    "points_count": collection_info.points_count,
                    "vector_size": collection_info.config.params.vectors.size,
                    "distance": collection_info.config.params.vectors.distance.value,
                }
            return {
                "qdrant_healthy": qdrant_healthy,
                "collection_exists": collection_exists,
                "collection_stats": collection_stats,
                "local_documents_count": len(self.documents),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return {
                "qdrant_healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
