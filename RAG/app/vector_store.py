"""
Vector Store Module
Handles storage and retrieval of document embeddings using ChromaDB
"""

import os
import json
import chromadb
from typing import List, Dict, Any, Tuple
from langchain.schema import Document
from app.config import config

class VectorStore:
    """Manages document embeddings storage and retrieval using ChromaDB"""
    
    def __init__(self, db_path: str = None):
        """
        Initialize vector store
        
        Args:
            db_path: Path to the vector database
        """
        self.db_path = db_path or config.VECTOR_DB_PATH
        self.client = None
        self.collection = None
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize ChromaDB client and collection"""
        try:
            print(f"🔄 Initializing vector database at: {self.db_path}")
            
            # Create directory if it doesn't exist
            os.makedirs(self.db_path, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path=self.db_path)
            
            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            
            print(f"✓ Successfully initialized vector database")
            print(f"✓ Collection: {self.collection.name}")
            
        except Exception as e:
            print(f"✗ Error initializing vector database: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> bool:
        """
        Add documents and their embeddings to the vector store
        
        Args:
            documents: List of Document objects
            embeddings: List of embedding vectors
            
        Returns:
            True if successful, False otherwise
        """
        if not self.collection:
            raise Exception("Vector store not initialized")
        
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match")
        
        try:
            # Prepare data for ChromaDB
            ids = []
            texts = []
            metadatas = []
            
            for i, doc in enumerate(documents):
                # Generate unique ID
                doc_id = f"doc_{i}_{hash(doc.page_content) % 1000000}"
                ids.append(doc_id)
                
                # Extract text content
                texts.append(doc.page_content)
                
                # Prepare metadata - ensure source is preserved
                metadata = doc.metadata.copy() if doc.metadata else {}
                metadata['chunk_id'] = doc_id
                metadata['chunk_index'] = i
                
                # Ensure source path is in metadata
                if 'source' not in metadata:
                    print(f"⚠️ Warning: No source found in document metadata for chunk {i}")
                
                metadatas.append(metadata)
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"✓ Successfully added {len(documents)} documents to vector store")
            print(f"✓ Total documents in store: {self.collection.count()}")
            return True
            
        except Exception as e:
            print(f"✗ Error adding documents to vector store: {str(e)}")
            return False
    
    def search_similar(self, query_embedding: List[float], 
                      top_k: int = None) -> List[Tuple[Document, float]]:
        """
        Search for similar documents based on query embedding
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of tuples (Document, similarity_score)
        """
        if not self.collection:
            raise Exception("Vector store not initialized")
        
        top_k = top_k or config.TOP_K_CHUNKS
        
        try:
            # First, check if we have any documents
            total_docs = self.collection.count()
            print(f"🔍 Searching in vector store with {total_docs} total documents")
            
            if total_docs == 0:
                print("⚠️ No documents found in vector store")
                return []
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, total_docs),
                include=['documents', 'metadatas', 'distances']
            )
            
            # Process results
            documents = []
            for i in range(len(results['documents'][0])):
                # Create Document object
                doc = Document(
                    page_content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i]
                )
                
                # Convert distance to similarity score (ChromaDB returns distances, not similarities)
                distance = results['distances'][0][i]
                similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
                
                # Only include documents with sufficient similarity (threshold from config)
                if similarity >= config.SIMILARITY_THRESHOLD:
                    documents.append((doc, similarity))
            
            print(f"✓ Found {len(documents)} similar documents above similarity threshold")
            return documents
            
        except Exception as e:
            print(f"✗ Error searching vector store: {str(e)}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection"""
        if not self.collection:
            return {}
        
        try:
            count = self.collection.count()
            return {
                'total_documents': count,
                'collection_name': self.collection.name,
                'database_path': self.db_path
            }
        except Exception as e:
            print(f"✗ Error getting collection stats: {str(e)}")
            return {}
    
    def has_documents(self) -> bool:
        """Check if the vector store has any documents"""
        if not self.collection:
            return False
        
        try:
            count = self.collection.count()
            return count > 0
        except Exception as e:
            print(f"✗ Error checking document count: {str(e)}")
            return False
    
    def get_all_documents(self) -> List[Document]:
        """Get all documents from the vector store (for debugging)"""
        if not self.collection:
            return []
        
        try:
            count = self.collection.count()
            if count == 0:
                return []
            
            # Get all documents
            results = self.collection.get(
                include=['documents', 'metadatas']
            )
            
            documents = []
            for i in range(len(results['documents'])):
                doc = Document(
                    page_content=results['documents'][i],
                    metadata=results['metadatas'][i]
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"✗ Error getting all documents: {str(e)}")
            return []
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection"""
        if not self.collection:
            return False
        
        try:
            self.collection.delete(where={})
            print("✓ Successfully cleared vector store collection")
            return True
        except Exception as e:
            print(f"✗ Error clearing collection: {str(e)}")
            return False
    
    def delete_documents_by_source(self, source_path: str) -> bool:
        """
        Delete documents from a specific source
        
        Args:
            source_path: Path of the source file
            
        Returns:
            True if successful, False otherwise
        """
        if not self.collection:
            return False
        
        try:
            # Delete documents with matching source
            self.collection.delete(where={"source": source_path})
            print(f"✓ Successfully deleted documents from source: {source_path}")
            return True
        except Exception as e:
            print(f"✗ Error deleting documents: {str(e)}")
            return False
    
    def get_document_by_id(self, doc_id: str) -> Document:
        """
        Retrieve a specific document by ID
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document object or None if not found
        """
        if not self.collection:
            return None
        
        try:
            results = self.collection.get(ids=[doc_id])
            if results['documents']:
                return Document(
                    page_content=results['documents'][0],
                    metadata=results['metadatas'][0] if results['metadatas'] else {}
                )
            return None
        except Exception as e:
            print(f"✗ Error retrieving document: {str(e)}")
            return None
