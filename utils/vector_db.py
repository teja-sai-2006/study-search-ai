import logging
import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
import tempfile

logger = logging.getLogger(__name__)

class VectorDatabase:
    """Vector database for semantic search using FAISS or Chroma"""
    
    def __init__(self, db_type: str = "faiss"):
        self.db_type = db_type
        self.index = None
        self.documents = []
        self.embeddings_model = None
        self.db_path = "vector_db"
        
        self._initialize_embeddings_model()
        self._initialize_database()
    
    def _initialize_embeddings_model(self):
        """Initialize sentence transformers model for embeddings"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Use a lightweight but effective model
            model_name = "all-MiniLM-L6-v2"
            self.embeddings_model = SentenceTransformer(model_name)
            logger.info(f"Initialized embeddings model: {model_name}")
            
        except ImportError:
            logger.error("sentence-transformers not installed")
            self.embeddings_model = None
        except Exception as e:
            logger.error(f"Failed to load embeddings model: {e}")
            self.embeddings_model = None
    
    def _initialize_database(self):
        """Initialize vector database"""
        if self.db_type == "faiss":
            self._initialize_faiss()
        elif self.db_type == "chroma":
            self._initialize_chroma()
        else:
            logger.error(f"Unsupported database type: {self.db_type}")
    
    def _initialize_faiss(self):
        """Initialize FAISS vector database"""
        try:
            import faiss
            
            # Create FAISS index (will be initialized when first document is added)
            self.faiss = faiss
            self.index = None
            logger.info("FAISS database initialized")
            
        except ImportError:
            logger.error("faiss-cpu not installed")
    
    def _initialize_chroma(self):
        """Initialize Chroma vector database"""
        try:
            import chromadb
            
            # Create persistent client
            self.chroma_client = chromadb.PersistentClient(path=self.db_path)
            self.collection = self.chroma_client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Chroma database initialized")
            
        except ImportError:
            logger.error("chromadb not installed")
    
    def add_documents(self, texts: List[str], metadata: List[Dict] = None) -> bool:
        """Add documents to vector database"""
        if not self.embeddings_model:
            logger.error("No embeddings model available")
            return False
        
        if not texts:
            return True
        
        try:
            # Generate embeddings
            embeddings = self.embeddings_model.encode(texts)
            
            if self.db_type == "faiss":
                return self._add_to_faiss(texts, embeddings, metadata)
            elif self.db_type == "chroma":
                return self._add_to_chroma(texts, embeddings, metadata)
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    def _add_to_faiss(self, texts: List[str], embeddings: np.ndarray, metadata: List[Dict] = None) -> bool:
        """Add documents to FAISS index"""
        try:
            if self.index is None:
                # Initialize index with first batch
                dimension = embeddings.shape[1]
                self.index = self.faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
                self.faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
            
            # Add to index
            self.index.add(embeddings.astype('float32'))
            
            # Store documents and metadata
            for i, text in enumerate(texts):
                doc_metadata = metadata[i] if metadata and i < len(metadata) else {}
                self.documents.append({
                    'text': text,
                    'metadata': doc_metadata,
                    'id': len(self.documents)
                })
            
            logger.info(f"Added {len(texts)} documents to FAISS index")
            return True
            
        except Exception as e:
            logger.error(f"FAISS add error: {e}")
            return False
    
    def _add_to_chroma(self, texts: List[str], embeddings: np.ndarray, metadata: List[Dict] = None) -> bool:
        """Add documents to Chroma collection"""
        try:
            # Prepare data for Chroma
            ids = [f"doc_{len(self.documents) + i}" for i in range(len(texts))]
            metadatas = metadata if metadata else [{}] * len(texts)
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            # Store locally for reference
            for i, text in enumerate(texts):
                doc_metadata = metadata[i] if metadata and i < len(metadata) else {}
                self.documents.append({
                    'text': text,
                    'metadata': doc_metadata,
                    'id': ids[i]
                })
            
            logger.info(f"Added {len(texts)} documents to Chroma collection")
            return True
            
        except Exception as e:
            logger.error(f"Chroma add error: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if not self.embeddings_model:
            logger.error("No embeddings model available")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embeddings_model.encode([query])
            
            if self.db_type == "faiss":
                return self._search_faiss(query_embedding, top_k)
            elif self.db_type == "chroma":
                return self._search_chroma(query, top_k)
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def _search_faiss(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Search FAISS index"""
        if self.index is None or len(self.documents) == 0:
            return []
        
        try:
            # Normalize query for cosine similarity
            self.faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding.astype('float32'), min(top_k, len(self.documents)))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    results.append({
                        'content': doc['text'],
                        'score': float(score),
                        'metadata': doc['metadata']
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"FAISS search error: {e}")
            return []
    
    def _search_chroma(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search Chroma collection"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, len(self.documents))
            )
            
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                    distance = results['distances'][0][i] if results['distances'] and results['distances'][0] else 0
                    
                    # Convert distance to similarity score (Chroma returns distances)
                    score = 1.0 - distance
                    
                    search_results.append({
                        'content': doc,
                        'score': score,
                        'metadata': metadata
                    })
            
            return search_results
            
        except Exception as e:
            logger.error(f"Chroma search error: {e}")
            return []
    
    def clear_database(self):
        """Clear all documents from database"""
        try:
            if self.db_type == "faiss":
                self.index = None
                self.documents = []
            elif self.db_type == "chroma":
                # Delete and recreate collection
                self.chroma_client.delete_collection("documents")
                self.collection = self.chroma_client.get_or_create_collection(
                    name="documents",
                    metadata={"hnsw:space": "cosine"}
                )
                self.documents = []
            
            logger.info("Vector database cleared")
            
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {
            'db_type': self.db_type,
            'total_documents': len(self.documents),
            'embeddings_model': self.embeddings_model.get_sentence_embedding_dimension() if self.embeddings_model else None,
            'index_initialized': self.index is not None
        }
    
    def save_database(self, path: str = None):
        """Save database to disk"""
        if not path:
            path = self.db_path
        
        try:
            os.makedirs(path, exist_ok=True)
            
            if self.db_type == "faiss" and self.index is not None:
                # Save FAISS index
                index_path = os.path.join(path, "faiss_index.bin")
                self.faiss.write_index(self.index, index_path)
                
                # Save documents metadata
                docs_path = os.path.join(path, "documents.json")
                with open(docs_path, 'w') as f:
                    json.dump(self.documents, f, indent=2)
                
                logger.info(f"FAISS database saved to {path}")
            
            # Chroma saves automatically in persistent mode
            
        except Exception as e:
            logger.error(f"Error saving database: {e}")
    
    def load_database(self, path: str = None):
        """Load database from disk"""
        if not path:
            path = self.db_path
        
        try:
            if self.db_type == "faiss":
                index_path = os.path.join(path, "faiss_index.bin")
                docs_path = os.path.join(path, "documents.json")
                
                if os.path.exists(index_path) and os.path.exists(docs_path):
                    # Load FAISS index
                    self.index = self.faiss.read_index(index_path)
                    
                    # Load documents
                    with open(docs_path, 'r') as f:
                        self.documents = json.load(f)
                    
                    logger.info(f"FAISS database loaded from {path}")
            
            # Chroma loads automatically in persistent mode
            
        except Exception as e:
            logger.error(f"Error loading database: {e}")
