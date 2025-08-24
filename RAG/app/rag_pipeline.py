"""
RAG Pipeline Module
Main orchestrator that combines all components for document processing and Q&A
"""

import os
import time
from typing import List, Dict, Any, Tuple
from langchain.schema import Document
from app.document_processor import DocumentProcessor
from app.embedding_generator import EmbeddingGenerator
from app.vector_store import VectorStore
from app.ollama_client import OllamaClient
from app.config import config

class RAGPipeline:
    """Main RAG pipeline that orchestrates all components"""
    
    def __init__(self):
        """Initialize RAG pipeline with all components"""
        print("🚀 Initializing RAG Pipeline...")
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore()
        self.ollama_client = OllamaClient()
        
        # Create necessary directories
        config.create_directories()
        
        print("✓ RAG Pipeline initialized successfully!")
    
    def process_and_store_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process a document and store it in the vector database
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        
        try:
            print(f"📄 Processing document: {os.path.basename(file_path)}")
            
            # Step 1: Process document into chunks
            chunks = self.document_processor.process_document(file_path)
            if not chunks:
                return {"success": False, "error": "No chunks generated"}
            
            # Step 2: Generate embeddings for chunks
            embeddings = self.embedding_generator.generate_embeddings(chunks)
            if len(embeddings) == 0:
                return {"success": False, "error": "No embeddings generated"}
            
            # Step 3: Store in vector database
            success = self.vector_store.add_documents(chunks, embeddings.tolist())
            if not success:
                return {"success": False, "error": "Failed to store in vector database"}
            
            processing_time = time.time() - start_time
            
            result = {
                "success": True,
                "filename": os.path.basename(file_path),
                "chunks_created": len(chunks),
                "processing_time": round(processing_time, 2),
                "file_size": os.path.getsize(file_path),
                "embedding_dimension": embeddings.shape[1]
            }
            
            print(f"✓ Document processed successfully in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"✗ Error processing document: {str(e)}")
            return {
                "success": False, 
                "error": str(e),
                "processing_time": round(processing_time, 2)
            }
    
    def answer_question(self, question: str, top_k: int = None) -> Dict[str, Any]:
        """
        Answer a question using RAG pipeline
        
        Args:
            question: User's question
            top_k: Number of top chunks to retrieve
            
        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        
        try:
            # Check if we have documents in the vector store
            if not self.vector_store.has_documents():
                return {
                    "success": False,
                    "error": "No documents found in the vector store. Please upload and process documents first.",
                    "processing_time": 0
                }
            
            print(f"🔍 Answering question: {question}")
            print(f"📚 Vector store has {self.vector_store.get_collection_stats().get('total_documents', 0)} documents")
            print(f"❓ Processing question: {question[:50]}...")
            
            # Step 1: Generate embedding for the question
            question_embedding = self.embedding_generator.generate_single_embedding(question)
            
            # Step 2: Search for similar documents
            similar_docs = self.vector_store.search_similar(
                question_embedding.tolist(), 
                top_k=top_k
            )
            
            if not similar_docs:
                return {
                    "success": False,
                    "answer": "I cannot answer this question based on the provided documents. No relevant information was found.",
                    "context": [],
                    "processing_time": round(time.time() - start_time, 2)
                }
            
            # Step 3: Prepare context from retrieved documents
            context_chunks = []
            for doc, similarity in similar_docs:
                context_chunks.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get('filename', 'Unknown'),
                    "similarity": round(similarity, 3),
                    "chunk_id": doc.metadata.get('chunk_id', 'Unknown')
                })
            
            # Additional check: ensure we have valid context chunks
            if not context_chunks:
                return {
                    "success": False,
                    "answer": "I cannot answer this question based on the provided documents. No relevant information was found.",
                    "context": [],
                    "processing_time": round(time.time() - start_time, 2)
                }
            
            # Step 4: Generate answer using Ollama with strict context-only instruction
            context_text = "\n\n".join([chunk["content"] for chunk in context_chunks])
            
            # Add additional instruction to ensure context-only answers
            enhanced_context = f"DOCUMENT CONTENT (ONLY USE THIS INFORMATION):\n{context_text}\n\nREMEMBER: Answer ONLY from the above document content."
            
            answer = self.ollama_client.generate_response(
                prompt=question,
                context=enhanced_context
            )
            
            processing_time = time.time() - start_time
            
            result = {
                "success": True,
                "answer": answer,
                "context": context_chunks,
                "processing_time": round(processing_time, 2),
                "chunks_retrieved": len(context_chunks),
                "question": question
            }
            
            print(f"✓ Question answered successfully in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"✗ Error answering question: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": round(processing_time, 2)
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics and status"""
        try:
            # Vector store stats
            vector_stats = self.vector_store.get_collection_stats()
            
            # Ollama models
            available_models = self.ollama_client.list_models()
            
            # Embedding model info
            embedding_dim = self.embedding_generator.get_embedding_dimension()
            
            stats = {
                "vector_database": vector_stats,
                "available_models": available_models,
                "embedding_dimension": embedding_dim,
                "current_model": self.ollama_client.model_name,
                "embedding_model": self.embedding_generator.model_name
            }
            
            return stats
            
        except Exception as e:
            print(f"✗ Error getting system stats: {str(e)}")
            return {"error": str(e)}
    
    def check_document_status(self) -> Dict[str, Any]:
        """Check the current document status in the pipeline"""
        try:
            stats = self.vector_store.get_collection_stats()
            has_docs = self.vector_store.has_documents()
            
            if has_docs:
                # Get a sample of documents for debugging
                sample_docs = self.vector_store.get_all_documents()[:3]
                sample_sources = [doc.metadata.get('source', 'Unknown') for doc in sample_docs]
                
                return {
                    "has_documents": True,
                    "total_documents": stats.get('total_documents', 0),
                    "sample_sources": sample_sources,
                    "status": "Ready for questions"
                }
            else:
                return {
                    "has_documents": False,
                    "total_documents": 0,
                    "sample_sources": [],
                    "status": "No documents found"
                }
                
        except Exception as e:
            return {
                "has_documents": False,
                "error": str(e),
                "status": "Error checking status"
            }
    
    def clear_all_documents(self) -> bool:
        """Clear all documents from the vector store"""
        try:
            success = self.vector_store.clear_collection()
            if success:
                print("✓ All documents cleared from vector store")
            return success
        except Exception as e:
            print(f"✗ Error clearing documents: {str(e)}")
            return False
    
    def delete_document(self, source_path: str) -> bool:
        """
        Delete a specific document and its chunks
        
        Args:
            source_path: Path of the source file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            success = self.vector_store.delete_documents_by_source(source_path)
            if success:
                print(f"✓ Document deleted: {os.path.basename(source_path)}")
            return success
        except Exception as e:
            print(f"✗ Error deleting document: {str(e)}")
            return False
    
    def batch_process_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple documents in batch
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            List of processing results for each document
        """
        results = []
        
        print(f"📚 Processing {len(file_paths)} documents in batch...")
        
        for i, file_path in enumerate(file_paths, 1):
            print(f"\n[{i}/{len(file_paths)}] Processing: {os.path.basename(file_path)}")
            result = self.process_and_store_document(file_path)
            results.append(result)
            
            # Add small delay between processing
            if i < len(file_paths):
                time.sleep(0.1)
        
        # Summary
        successful = sum(1 for r in results if r["success"])
        total_time = sum(r.get("processing_time", 0) for r in results)
        
        print(f"\n📊 Batch processing complete!")
        print(f"  Successful: {successful}/{len(file_paths)}")
        print(f"  Total time: {total_time:.2f}s")
        
        return results

    def switch_model(self, new_model: str) -> bool:
        """
        Switch to a different LLM model
        
        Args:
            new_model: Name of the new model to use
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # First verify the model is available
            available_models = self.get_available_models()
            if new_model not in available_models:
                print(f"✗ Model '{new_model}' not found in available models: {available_models}")
                return False
            
            # Switch the model
            success = self.ollama_client.switch_model(new_model)
            if success:
                print(f"✓ RAG Pipeline switched to model: {new_model}")
                # Update the config as well
                config.update_model_selection(new_model)
            return success
        except Exception as e:
            print(f"✗ Error switching model in RAG pipeline: {str(e)}")
            return False
    
    def get_current_model(self) -> str:
        """Get the currently selected model name"""
        return self.ollama_client.get_current_model()
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama"""
        try:
            models = self.ollama_client.list_models()
            # Extract just the model names from the response
            model_names = [model.get('name', '') for model in models if model.get('name')]
            print(f"✓ Available models detected: {model_names}")
            return model_names
        except Exception as e:
            print(f"✗ Error getting available models: {str(e)}")
            # Fallback to config if Ollama is not accessible
            return config.AVAILABLE_MODELS
