from langchain_openai import AzureOpenAIEmbeddings
import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from core.config import Config
from utils.custom_logger import CustomLogger


class PolicyEmbeddings:
    def __init__(
        self,
        qdrant_host="localhost",
        qdrant_port=6333,
        collection_name="policy_documents",
        storage_path="logs/policy_embeddings",
    ):
        self.logger = CustomLogger("PolicyEmbeddings")
        self.storage_path = storage_path
        self.collection_name = collection_name
        self.documents = []

        # Create storage directory for metadata
        os.makedirs(storage_path, exist_ok=True)

        # File path for metadata storage
        self.metadata_path = os.path.join(storage_path, "metadata.json")

        # Initialize Qdrant client
        try:
            self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
            self.logger.info(
                f"Successfully connected to Qdrant at {qdrant_host}:{qdrant_port}"
            )

            # Test connection
            collections = self.qdrant_client.get_collections()
            self.logger.debug(
                f"Available collections: {[c.name for c in collections.collections]}"
            )

        except Exception as e:
            self.logger.error(f"Failed to connect to Qdrant: {str(e)}")
            self.logger.error("Make sure Qdrant is running in Docker:")
            self.logger.error("docker run -p 6333:6333 qdrant/qdrant")
            raise

        # Initialize OpenAI client
        try:
            self.client = AzureOpenAIEmbeddings(
                openai_api_version=Config.DIAL_API_VERSION,
                azure_deployment=Config.EMBEDDING_MODEL_NAME,
                azure_endpoint=Config.DIAL_API_ENDPOINT,
                api_key=Config.DIAL_API_KEY,
                # Set the flag to False for models which do not support token ids in inputs
                check_embedding_ctx_length=False,
            )
            self.logger.info("Successfully initialized OpenAI client")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise

    def _calculate_content_hash(self, content: str) -> str:
        """Calculate SHA-256 hash of content"""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _calculate_directory_hash(self, docs_path: str) -> str:
        """Calculate hash of all files in directory to detect changes"""
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
        """Load metadata from JSON file"""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load metadata: {str(e)}")
        return {}

    def _save_metadata(self, metadata: Dict[str, Any]):
        """Save metadata to JSON file"""
        try:
            with open(self.metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {str(e)}")
            raise

    def _has_policy_changed(self, docs_path: str) -> tuple[bool, str]:
        """Check if policies have changed since last embedding generation"""
        current_hash = self._calculate_directory_hash(docs_path)
        metadata = self._load_metadata()

        stored_hash = metadata.get("policy_versions", {}).get(docs_path)

        if stored_hash is None:
            # First time processing this directory
            return True, current_hash

        return stored_hash != current_hash, current_hash

    def _update_policy_version(self, docs_path: str, version_hash: str):
        """Update the stored policy version hash"""
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

    def _ensure_collection_exists(self):
        """Ensure the Qdrant collection exists with proper configuration"""
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                self.logger.info(f"Creating collection: {self.collection_name}")

                # Create collection with OpenAI embedding dimensions (1536 for text-embedding-ada-002)
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=1536,  # OpenAI text-embedding-ada-002 dimension
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

    def _load_documents_from_qdrant(self) -> List[Dict[str, Any]]:
        """Load existing documents from Qdrant collection"""
        try:
            # Get all points from the collection
            response = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Adjust based on your needs
                with_payload=True,
                with_vectors=False,
            )

            documents = []
            for point in response[0]:  # response is (points, next_page_offset)
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
        """Load documents and check if embeddings need to be regenerated"""
        self.logger.info(f"Loading documents from {docs_path}")

        # Ensure collection exists
        self._ensure_collection_exists()

        # Check if policies have changed
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

            # Clear existing collection
            try:
                self.qdrant_client.delete_collection(self.collection_name)
                self.logger.info(f"Cleared existing collection: {self.collection_name}")
            except:
                pass  # Collection might not exist

            # Recreate collection
            self._ensure_collection_exists()

            # Load and process documents
            documents = []
            doc_counter = 0

            for filename in os.listdir(docs_path):
                if filename.endswith(".txt"):
                    filepath = os.path.join(docs_path, filename)
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        content_hash = self._calculate_content_hash(content)

                        chunks = self._chunk_text(content, max_length=1000)
                        for i, chunk in enumerate(chunks):
                            documents.append(
                                {
                                    "content": chunk,
                                    "source": filename,
                                    "chunk_id": i,
                                    "content_hash": content_hash,
                                    "doc_id": doc_counter,
                                }
                            )
                            doc_counter += 1

            self.documents = documents
            self.logger.info(f"Loaded {len(documents)} document chunks")

            # Update version tracking
            self._update_policy_version(docs_path, current_hash)

        return self.documents

    def _chunk_text(self, text: str, max_length: int = 1000) -> List[str]:
        """Split text into smaller chunks"""
        sentences = text.split(". ")
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk + sentence) < max_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def create_embeddings(self):
        """Create embeddings for all documents using OpenAI and store in Qdrant"""
        try:
            if not self.documents:
                raise ValueError("No documents loaded. Call load_documents() first.")

            # Check if documents are already in Qdrant
            existing_docs = self._load_documents_from_qdrant()
            if len(existing_docs) == len(self.documents):
                self.logger.info("Embeddings already exist in Qdrant")
                return

            self.logger.info("Creating embeddings for documents")
            texts = [doc["content"] for doc in self.documents]

            # Process in batches to handle API limits
            batch_size = 100
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                self.logger.debug(
                    f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}"
                )

                # Use embed_documents for batch embedding (returns list of vectors)
                response = self.client.embed_documents(batch)

                all_embeddings.extend(response)

            # Prepare points for Qdrant
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

            # Upload to Qdrant in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                self.qdrant_client.upsert(
                    collection_name=self.collection_name, points=batch
                )
                self.logger.debug(f"Uploaded batch {i//batch_size + 1} to Qdrant")

            self.logger.info(f"Created and stored {len(points)} embeddings in Qdrant")

        except Exception as e:
            self.logger.error(f"Failed to create embeddings: {str(e)}")
            raise

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for most relevant documents using Qdrant"""
        try:
            self.logger.debug(f"Searching for query: {query[:50]}...")

            # Get query embedding from OpenAI
            query_embedding = self.client.embed_query(query)
            # query_embedding is a list of floats

            # Search using Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True,
            )

            results = []
            for i, result in enumerate(search_results):
                results.append(
                    {
                        "content": result.payload["content"],
                        "source": result.payload["source"],
                        "chunk_id": result.payload["chunk_id"],
                        "score": round(
                            float(result.score) * 100, 2
                        ),  # Match score as percentage
                        "rank": i + 1,
                    }
                )

            self.logger.info(f"Found {len(results)} matching documents")
            return results

        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            raise

    def add_document(
        self, content: str, source: str, save_immediately: bool = True
    ) -> int:
        """Add a single document and update the Qdrant collection"""
        try:
            self.logger.info(f"Adding new document from {source}")

            # Process the new document
            content_hash = self._calculate_content_hash(content)
            chunks = self._chunk_text(content, max_length=1000)

            # Get current max doc_id
            max_doc_id = max([doc["doc_id"] for doc in self.documents] + [-1])

            new_docs = []
            for i, chunk in enumerate(chunks):
                new_doc = {
                    "content": chunk,
                    "source": source,
                    "chunk_id": i,
                    "content_hash": content_hash,
                    "doc_id": max_doc_id + 1 + i,
                }
                new_docs.append(new_doc)

            # Generate embeddings for new documents
            texts = [doc["content"] for doc in new_docs]
            embeddings = self.client.embed_query("".join(texts))

            # Prepare points for Qdrant
            points = []
            for i, (doc, embedding) in enumerate(zip(new_docs, embeddings)):
                # Get next available point ID
                existing_points = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    limit=1,
                    with_payload=False,
                    with_vectors=False,
                )
                next_id = len(existing_points[0]) if existing_points[0] else 0

                point = PointStruct(
                    id=next_id + i,
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

            # Add to Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name, points=points
            )

            # Update local documents list
            self.documents.extend(new_docs)

            self.logger.info(f"Added {len(new_docs)} chunks from {source}")
            return len(new_docs)

        except Exception as e:
            self.logger.error(f"Failed to add document: {str(e)}")
            raise

    def remove_documents_by_source(
        self, source: str, save_immediately: bool = True
    ) -> int:
        """Remove all documents from a specific source"""
        try:
            self.logger.info(f"Removing documents from source: {source}")

            # Find documents to remove
            docs_to_remove = [doc for doc in self.documents if doc["source"] == source]

            if not docs_to_remove:
                self.logger.warning(f"No documents found for source: {source}")
                return 0

            # Remove from Qdrant using filter
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="source", match=MatchValue(value=source))]
                ),
            )

            # Update local documents list
            self.documents = [doc for doc in self.documents if doc["source"] != source]

            self.logger.info(f"Removed {len(docs_to_remove)} chunks from {source}")
            return len(docs_to_remove)

        except Exception as e:
            self.logger.error(f"Failed to remove documents: {str(e)}")
            raise

    def force_refresh(self, docs_path: str):
        """Force refresh of embeddings regardless of whether policies have changed"""
        self.logger.info("Forcing refresh of embeddings")

        try:
            # Clear Qdrant collection
            try:
                self.qdrant_client.delete_collection(self.collection_name)
                self.logger.info(f"Deleted collection: {self.collection_name}")
            except:
                pass  # Collection might not exist

            # Clear metadata for this path
            metadata = self._load_metadata()
            if (
                "policy_versions" in metadata
                and docs_path in metadata["policy_versions"]
            ):
                del metadata["policy_versions"][docs_path]
                self._save_metadata(metadata)

            # Reset internal state
            self.documents = []

            # Reload everything
            self.load_documents(docs_path)
            self.create_embeddings()

        except Exception as e:
            self.logger.error(f"Error during force refresh: {str(e)}")
            raise

    def get_storage_stats(self) -> Optional[Dict[str, Any]]:
        """Get statistics about the Qdrant storage"""
        try:
            metadata = self._load_metadata()

            # Get collection info from Qdrant
            try:
                collection_info = self.qdrant_client.get_collection(
                    self.collection_name
                )
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
                "qdrant_connection": {
                    "host": (
                        self.qdrant_client._client.host
                        if hasattr(self.qdrant_client, "_client")
                        else "unknown"
                    ),
                    "port": (
                        self.qdrant_client._client.port
                        if hasattr(self.qdrant_client, "_client")
                        else "unknown"
                    ),
                },
            }

            # Calculate metadata file size
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

    def optimize_index(self):
        """Optimize the Qdrant collection for better search performance"""
        try:
            # Qdrant automatically optimizes its indexes, but we can trigger optimization
            self.logger.info("Triggering Qdrant collection optimization")

            # Get collection info
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            current_size = collection_info.points_count

            if current_size > 1000:
                self.logger.info(
                    f"Collection has {current_size} points, optimization may improve performance"
                )

                # For large collections, you might want to configure HNSW parameters
                # This would require recreating the collection with optimized settings
                # For now, we'll just log the recommendation
                self.logger.info(
                    "Consider configuring HNSW parameters for large collections"
                )
                self.logger.info(
                    "Qdrant automatically handles index optimization in the background"
                )
            else:
                self.logger.info("Collection size doesn't require special optimization")

        except Exception as e:
            self.logger.error(f"Failed to optimize collection: {str(e)}")
            raise

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on Qdrant connection and collection"""
        try:
            # Check Qdrant connection
            collections = self.qdrant_client.get_collections()
            qdrant_healthy = True

            # Check if our collection exists
            collection_names = [c.name for c in collections.collections]
            collection_exists = self.collection_name in collection_names

            # Get collection stats if it exists
            collection_stats = None
            if collection_exists:
                collection_info = self.qdrant_client.get_collection(
                    self.collection_name
                )
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
