#!/usr/bin/env python3
"""
Test script to verify vector store functionality
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.vector_store import VectorStore
from app.rag_pipeline import RAGPipeline
from app.config import config

def test_vector_store():
    """Test basic vector store functionality"""
    print("🧪 Testing Vector Store...")
    
    try:
        # Initialize vector store
        vector_store = VectorStore()
        print("✅ Vector store initialized")
        
        # Check initial status
        stats = vector_store.get_collection_stats()
        print(f"📊 Initial stats: {stats}")
        
        # Check if documents exist
        has_docs = vector_store.has_documents()
        print(f"📚 Has documents: {has_docs}")
        
        if has_docs:
            # Get all documents
            docs = vector_store.get_all_documents()
            print(f"📄 Found {len(docs)} documents:")
            for i, doc in enumerate(docs[:3]):  # Show first 3
                print(f"  {i+1}. Source: {doc.metadata.get('source', 'Unknown')}")
                print(f"     Content preview: {doc.page_content[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing vector store: {str(e)}")
        return False

def test_rag_pipeline():
    """Test RAG pipeline document status"""
    print("\n🧪 Testing RAG Pipeline...")
    
    try:
        # Initialize pipeline
        pipeline = RAGPipeline()
        print("✅ RAG pipeline initialized")
        
        # Check document status
        doc_status = pipeline.check_document_status()
        print(f"📚 Document status: {doc_status}")
        
        # Check system stats
        stats = pipeline.get_system_stats()
        print(f"📊 System stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing RAG pipeline: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Vector Store Tests...")
    print("=" * 50)
    
    # Test vector store
    vs_success = test_vector_store()
    
    # Test RAG pipeline
    rag_success = test_rag_pipeline()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"  Vector Store: {'✅ PASS' if vs_success else '❌ FAIL'}")
    print(f"  RAG Pipeline: {'✅ PASS' if rag_success else '❌ FAIL'}")
    
    if vs_success and rag_success:
        print("\n🎉 All tests passed! Vector store is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()
