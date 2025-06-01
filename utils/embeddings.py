from openai import AzureOpenAI
import numpy as np
import faiss
import os
import pickle
import hashlib
import json
from datetime import datetime
from core.config import Config
from utils.custom_logger import CustomLogger

class PolicyEmbeddings:
    def __init__(self, storage_path="embeddings_storage"):
        self.logger = CustomLogger("PolicyEmbeddings")
        self.storage_path = storage_path
        self.index = None
        self.documents = []
        self.embeddings = None
        
        # Create storage directory
        os.makedirs(storage_path, exist_ok=True)
        
        # File paths for persistent storage
        self.index_path = os.path.join(storage_path, "faiss_index.bin")
        self.documents_path = os.path.join(storage_path, "documents.pkl")
        self.metadata_path = os.path.join(storage_path, "metadata.json")
        self.embeddings_path = os.path.join(storage_path, "embeddings.npy")
        
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

    def _calculate_content_hash(self, content):
        """Calculate SHA-256 hash of content"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _calculate_directory_hash(self, docs_path):
        """Calculate hash of all files in directory to detect changes"""
        file_hashes = []
        if os.path.exists(docs_path):
            for filename in sorted(os.listdir(docs_path)):
                if filename.endswith('.txt'):
                    filepath = os.path.join(docs_path, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        file_hashes.append(f"{filename}:{self._calculate_content_hash(content)}")
        
        combined_hash = hashlib.sha256('|'.join(file_hashes).encode('utf-8')).hexdigest()
        return combined_hash

    def _load_metadata(self):
        """Load metadata from JSON file"""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load metadata: {str(e)}")
        return {}

    def _save_metadata(self, metadata):
        """Save metadata to JSON file"""
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {str(e)}")
            raise

    def _has_policy_changed(self, docs_path):
        """Check if policies have changed since last embedding generation"""
        current_hash = self._calculate_directory_hash(docs_path)
        metadata = self._load_metadata()
        
        stored_hash = metadata.get('policy_versions', {}).get(docs_path)
        
        if stored_hash is None:
            # First time processing this directory
            return True, current_hash
        
        return stored_hash != current_hash, current_hash

    def _update_policy_version(self, docs_path, version_hash):
        """Update the stored policy version hash"""
        try:
            metadata = self._load_metadata()
            
            if 'policy_versions' not in metadata:
                metadata['policy_versions'] = {}
            
            metadata['policy_versions'][docs_path] = version_hash
            metadata['last_updated'] = datetime.now().isoformat()
            
            self._save_metadata(metadata)
            
        except Exception as e:
            self.logger.error(f"Error updating policy version: {str(e)}")
            raise

    def _save_to_faiss_storage(self):
        """Save FAISS index, documents, and embeddings to disk"""
        try:
            # Save FAISS index
            if self.index is not None:
                faiss.write_index(self.index, self.index_path)
                self.logger.debug("FAISS index saved successfully")
            
            # Save documents
            with open(self.documents_path, 'wb') as f:
                pickle.dump(self.documents, f)
                self.logger.debug("Documents saved successfully")
            
            # Save embeddings as numpy array
            if self.embeddings is not None:
                np.save(self.embeddings_path, self.embeddings)
                self.logger.debug("Embeddings saved successfully")
            
            self.logger.info("All data saved to FAISS storage successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving to FAISS storage: {str(e)}")
            raise

    def _load_from_faiss_storage(self):
        """Load FAISS index, documents, and embeddings from disk"""
        try:
            # Check if all required files exist
            required_files = [self.index_path, self.documents_path, self.embeddings_path]
            if not all(os.path.exists(f) for f in required_files):
                self.logger.warning("Some storage files missing, cannot load from storage")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(self.index_path)
            self.logger.debug("FAISS index loaded successfully")
            
            # Load documents
            with open(self.documents_path, 'rb') as f:
                self.documents = pickle.load(f)
                self.logger.debug("Documents loaded successfully")
            
            # Load embeddings
            self.embeddings = np.load(self.embeddings_path)
            self.logger.debug("Embeddings loaded successfully")
            
            self.logger.info(f"Loaded {len(self.documents)} documents and embeddings from FAISS storage")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading from FAISS storage: {str(e)}")
            return False

    def load_documents(self, docs_path):
        """Load documents and check if embeddings need to be regenerated"""
        self.logger.info(f"Loading documents from {docs_path}")
        
        # Check if policies have changed
        has_changed, current_hash = self._has_policy_changed(docs_path)
        
        if not has_changed:
            self.logger.info("No policy changes detected, loading existing embeddings")
            if self._load_from_faiss_storage():
                return self.documents
            else:
                self.logger.warning("Failed to load from storage, will regenerate embeddings")
                has_changed = True
        
        if has_changed:
            self.logger.info("Policy changes detected or storage unavailable, processing documents")
            
            # Load and process documents
            documents = []
            for filename in os.listdir(docs_path):
                if filename.endswith('.txt'):
                    filepath = os.path.join(docs_path, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        content_hash = self._calculate_content_hash(content)
                        
                        chunks = self._chunk_text(content, max_length=200)
                        for i, chunk in enumerate(chunks):
                            documents.append({
                                'content': chunk,
                                'source': filename,
                                'chunk_id': i,
                                'content_hash': content_hash,
                                'doc_id': len(documents)
                            })
            
            self.documents = documents
            self.logger.info(f"Loaded {len(documents)} document chunks")
            
            # Update version tracking
            self._update_policy_version(docs_path, current_hash)
        
        return self.documents

    def _chunk_text(self, text, max_length=200):
        """Split text into smaller chunks"""
        sentences = text.split('. ')
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
        """Create embeddings for all documents using Azure OpenAI and store in FAISS"""
        try:
            # Check if embeddings already exist and are up to date
            if self.embeddings is not None and self.index is not None:
                self.logger.info("Embeddings already loaded and up to date")
                return
            
            if not self.documents:
                raise ValueError("No documents loaded. Call load_documents() first.")
            
            self.logger.info("Creating embeddings for documents")
            texts = [doc['content'] for doc in self.documents]
            
            # Process in batches to handle API limits
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                self.logger.debug(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")
                
                response = self.client.embeddings.create(
                    input=batch,
                    model="text-embedding-ada-002"
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            
            # Convert to numpy array and ensure correct format
            self.embeddings = np.array(all_embeddings, dtype=np.float32)
            self.embeddings = np.ascontiguousarray(self.embeddings)
            
            # Create FAISS index
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            
            # Normalize embeddings for cosine similarity
            embeddings_normalized = self.embeddings.copy()
            faiss.normalize_L2(embeddings_normalized)
            self.index.add(embeddings_normalized)
            
            # Save everything to disk
            self._save_to_faiss_storage()
            
            self.logger.info(f"Created and stored embeddings with shape {self.embeddings.shape}")
            
        except Exception as e:
            self.logger.error(f"Failed to create embeddings: {str(e)}")
            raise

    def search(self, query, top_k=3):
        """Search for most relevant documents using FAISS"""
        try:
            if self.index is None:
                raise ValueError("Index not created. Call create_embeddings() first.")
            
            self.logger.debug(f"Searching for query: {query[:50]}...")
            
            # Get query embedding from Azure OpenAI
            response = self.client.embeddings.create(
                input=query,
                model="text-embedding-ada-002"
            )
            query_embedding = np.array([response.data[0].embedding], dtype=np.float32)
            query_embedding = np.ascontiguousarray(query_embedding)
            
            # Normalize query embedding for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search using FAISS
            scores, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1:  # Valid index
                    results.append({
                        'content': self.documents[idx]['content'],
                        'source': self.documents[idx]['source'],
                        'chunk_id': self.documents[idx]['chunk_id'],
                        'score': float(score),
                        'rank': i + 1
                    })
            
            self.logger.info(f"Found {len(results)} matching documents")
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            raise

    def add_document(self, content, source, save_immediately=True):
        """Add a single document and update the FAISS index"""
        try:
            self.logger.info(f"Adding new document from {source}")
            
            # Process the new document
            content_hash = self._calculate_content_hash(content)
            chunks = self._chunk_text(content, max_length=200)
            
            new_docs = []
            for i, chunk in enumerate(chunks):
                new_doc = {
                    'content': chunk,
                    'source': source,
                    'chunk_id': i,
                    'content_hash': content_hash,
                    'doc_id': len(self.documents) + len(new_docs)
                }
                new_docs.append(new_doc)
            
            # Generate embeddings for new documents
            texts = [doc['content'] for doc in new_docs]
            response = self.client.embeddings.create(
                input=texts,
                model="text-embedding-ada-002"
            )
            
            new_embeddings = np.array([item.embedding for item in response.data], dtype=np.float32)
            new_embeddings = np.ascontiguousarray(new_embeddings)
            
            # Update documents list
            self.documents.extend(new_docs)
            
            # Update embeddings array
            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
            
            # Update FAISS index
            if self.index is None:
                dimension = new_embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)
            
            # Normalize and add to index
            new_embeddings_normalized = new_embeddings.copy()
            faiss.normalize_L2(new_embeddings_normalized)
            self.index.add(new_embeddings_normalized)
            
            if save_immediately:
                self._save_to_faiss_storage()
            
            self.logger.info(f"Added {len(new_docs)} chunks from {source}")
            return len(new_docs)
            
        except Exception as e:
            self.logger.error(f"Failed to add document: {str(e)}")
            raise

    def remove_documents_by_source(self, source, save_immediately=True):
        """Remove all documents from a specific source"""
        try:
            self.logger.info(f"Removing documents from source: {source}")
            
            # Find indices of documents to remove
            indices_to_remove = [i for i, doc in enumerate(self.documents) if doc['source'] == source]
            
            if not indices_to_remove:
                self.logger.warning(f"No documents found for source: {source}")
                return 0
            
            # Remove from documents list (in reverse order to maintain indices)
            for idx in reversed(indices_to_remove):
                del self.documents[idx]
            
            # Rebuild the entire index (FAISS doesn't support efficient removal)
            self._rebuild_index()
            
            if save_immediately:
                self._save_to_faiss_storage()
            
            self.logger.info(f"Removed {len(indices_to_remove)} chunks from {source}")
            return len(indices_to_remove)
            
        except Exception as e:
            self.logger.error(f"Failed to remove documents: {str(e)}")
            raise

    def _rebuild_index(self):
        """Rebuild the FAISS index from current documents"""
        if not self.documents:
            self.index = None
            self.embeddings = None
            return
        
        try:
            # Regenerate embeddings for all documents
            texts = [doc['content'] for doc in self.documents]
            
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = self.client.embeddings.create(
                    input=batch,
                    model="text-embedding-ada-002"
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            
            # Update embeddings and index
            self.embeddings = np.array(all_embeddings, dtype=np.float32)
            self.embeddings = np.ascontiguousarray(self.embeddings)
            
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            
            embeddings_normalized = self.embeddings.copy()
            faiss.normalize_L2(embeddings_normalized)
            self.index.add(embeddings_normalized)
            
            self.logger.info("FAISS index rebuilt successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to rebuild index: {str(e)}")
            raise

    def force_refresh(self, docs_path):
        """Force refresh of embeddings regardless of whether policies have changed"""
        self.logger.info("Forcing refresh of embeddings")
        
        try:
            # Clear storage files
            storage_files = [self.index_path, self.documents_path, self.embeddings_path]
            for file_path in storage_files:
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            # Clear metadata for this path
            metadata = self._load_metadata()
            if 'policy_versions' in metadata and docs_path in metadata['policy_versions']:
                del metadata['policy_versions'][docs_path]
                self._save_metadata(metadata)
            
            # Reset internal state
            self.index = None
            self.documents = []
            self.embeddings = None
            
            # Reload everything
            self.load_documents(docs_path)
            self.create_embeddings()
            
        except Exception as e:
            self.logger.error(f"Error during force refresh: {str(e)}")
            raise

    def get_storage_stats(self):
        """Get statistics about the FAISS storage"""
        try:
            metadata = self._load_metadata()
            
            stats = {
                'document_count': len(self.documents) if self.documents else 0,
                'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0,
                'index_size': self.index.ntotal if self.index else 0,
                'storage_path': self.storage_path,
                'policy_versions': metadata.get('policy_versions', {}),
                'last_updated': metadata.get('last_updated'),
                'storage_files': {
                    'index_exists': os.path.exists(self.index_path),
                    'documents_exists': os.path.exists(self.documents_path),
                    'embeddings_exists': os.path.exists(self.embeddings_path),
                    'metadata_exists': os.path.exists(self.metadata_path)
                }
            }
            
            # Calculate storage sizes
            for file_key, file_path in [
                ('index_size_mb', self.index_path),
                ('documents_size_mb', self.documents_path),
                ('embeddings_size_mb', self.embeddings_path),
                ('metadata_size_mb', self.metadata_path)
            ]:
                if os.path.exists(file_path):
                    stats[file_key] = round(os.path.getsize(file_path) / (1024 * 1024), 2)
                else:
                    stats[file_key] = 0
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting storage stats: {str(e)}")
            return None

    def optimize_index(self):
        """Optimize the FAISS index for better search performance"""
        try:
            if self.index is None or self.embeddings is None:
                self.logger.warning("No index to optimize")
                return
            
            # For larger datasets, you might want to use more advanced FAISS indices
            # like IndexIVFFlat or IndexHNSWFlat for better performance
            current_size = self.index.ntotal
            
            if current_size > 10000:  # For larger datasets
                self.logger.info("Optimizing index for large dataset")
                
                # Create an IVF index for better performance on large datasets
                quantizer = faiss.IndexFlatIP(self.embeddings.shape[1])
                nlist = min(100, max(1, current_size // 100))  # Number of clusters
                index_ivf = faiss.IndexIVFFlat(quantizer, self.embeddings.shape[1], nlist)
                
                # Train the index
                embeddings_normalized = self.embeddings.copy()
                faiss.normalize_L2(embeddings_normalized)
                index_ivf.train(embeddings_normalized)
                index_ivf.add(embeddings_normalized)
                
                # Replace the current index
                self.index = index_ivf
                self.logger.info(f"Index optimized with IVF, nlist={nlist}")
                
                # Save the optimized index
                self._save_to_faiss_storage()
            else:
                self.logger.info("Dataset size doesn't require optimization")
                
        except Exception as e:
            self.logger.error(f"Failed to optimize index: {str(e)}")
            raise