"""
Embedding Generator Module
Uses sentence-transformers to generate embeddings for document chunks
"""

import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from app.config import config

class EmbeddingGenerator:
    """Generates embeddings for document chunks using sentence-transformers"""
    
    def __init__(self, model_name: str = None):
        """
        Initialize embedding generator
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name or config.EMBEDDING_MODEL_NAME
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence-transformers model"""
        try:
            print(f"🔄 Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"✓ Successfully loaded embedding model: {self.model_name}")
        except Exception as e:
            print(f"✗ Error loading embedding model: {str(e)}")
            # Fallback to default model
            try:
                print("🔄 Trying fallback model: all-MiniLM-L6-v2")
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✓ Successfully loaded fallback model")
            except Exception as fallback_error:
                raise Exception(f"Failed to load any embedding model: {str(fallback_error)}")
    
    def generate_embeddings(self, texts: Union[List[str], List[Document]]) -> np.ndarray:
        """
        Generate embeddings for a list of texts or documents
        
        Args:
            texts: List of text strings or Document objects
            
        Returns:
            Numpy array of embeddings
        """
        if not self.model:
            raise Exception("Embedding model not loaded")
        
        # Extract text content if documents are provided
        if texts and isinstance(texts[0], Document):
            text_list = [doc.page_content for doc in texts]
        else:
            text_list = texts
        
        if not text_list:
            return np.array([])
        
        try:
            print(f"🔄 Generating embeddings for {len(text_list)} texts...")
            embeddings = self.model.encode(text_list, show_progress_bar=True)
            print(f"✓ Generated embeddings with shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            print(f"✗ Error generating embeddings: {str(e)}")
            raise
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text string
            
        Returns:
            Numpy array of single embedding
        """
        if not self.model:
            raise Exception("Embedding model not loaded")
        
        try:
            embedding = self.model.encode([text])
            return embedding[0]  # Return single embedding
        except Exception as e:
            print(f"✗ Error generating single embedding: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings"""
        if not self.model:
            raise Exception("Embedding model not loaded")
        
        # Generate a test embedding to get dimension
        test_embedding = self.generate_single_embedding("test")
        return len(test_embedding)
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def batch_similarity(self, query_embedding: np.ndarray, 
                        document_embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate similarity between query and multiple document embeddings
        
        Args:
            query_embedding: Query embedding vector
            document_embeddings: Array of document embeddings
            
        Returns:
            Array of similarity scores
        """
        if not self.model:
            raise Exception("Embedding model not loaded")
        
        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return np.zeros(len(document_embeddings))
        
        # Normalize document embeddings
        doc_norms = np.linalg.norm(document_embeddings, axis=1)
        
        # Calculate cosine similarities
        similarities = np.dot(document_embeddings, query_embedding) / (doc_norms * query_norm)
        
        # Handle division by zero
        similarities = np.nan_to_num(similarities, nan=0.0)
        
        return similarities
