"""
Test Setup Script
Run this to verify all components are working correctly
"""

import os
import sys
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        # Test core modules
        from app.config import config
        print("✅ Config module imported successfully")
        
        from app.document_processor import DocumentProcessor
        print("✅ Document processor imported successfully")
        
        from app.embedding_generator import EmbeddingGenerator
        print("✅ Embedding generator imported successfully")
        
        from app.vector_store import VectorStore
        print("✅ Vector store imported successfully")
        
        from app.ollama_client import OllamaClient
        print("✅ Ollama client imported successfully")
        
        from app.rag_pipeline import RAGPipeline
        print("✅ RAG pipeline imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_configuration():
    """Test configuration loading"""
    print("\n⚙️ Testing configuration...")
    
    try:
        from app.config import config
        
        # Check if config loaded
        print(f"✅ Ollama URL: {config.OLLAMA_BASE_URL}")
        print(f"✅ Model: {config.OLLAMA_MODEL_NAME}")
        print(f"✅ Embedding Model: {config.EMBEDDING_MODEL_NAME}")
        print(f"✅ Chunk Size: {config.CHUNK_SIZE}")
        print(f"✅ Vector DB Path: {config.VECTOR_DB_PATH}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_directories():
    """Test if required directories exist"""
    print("\n📁 Testing directories...")
    
    try:
        from app.config import config
        
        # Create directories
        config.create_directories()
        
        # Check if they exist
        required_dirs = [
            "data/documents",
            "data/uploaded",
            "data/vector_db"
        ]
        
        for dir_path in required_dirs:
            if Path(dir_path).exists():
                print(f"✅ Directory exists: {dir_path}")
            else:
                print(f"❌ Directory missing: {dir_path}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Directory error: {e}")
        return False

def test_ollama_connection():
    """Test Ollama server connection"""
    print("\n🔌 Testing Ollama connection...")
    
    try:
        from app.ollama_client import OllamaClient
        
        # This will test connection
        client = OllamaClient()
        print("✅ Ollama connection successful")
        
        # Test model listing
        models = client.list_models()
        print(f"✅ Available models: {len(models)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        print("💡 Make sure Ollama is running: ollama serve")
        return False

def test_embedding_model():
    """Test embedding model loading"""
    print("\n🧠 Testing embedding model...")
    
    try:
        from app.embedding_generator import EmbeddingGenerator
        
        # This will download and load the model
        generator = EmbeddingGenerator()
        
        # Test embedding generation
        test_text = "This is a test sentence."
        embedding = generator.generate_single_embedding(test_text)
        
        print(f"✅ Embedding model loaded successfully")
        print(f"✅ Embedding dimension: {len(embedding)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding model error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 RAG Chatbot Setup Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_configuration),
        ("Directory Test", test_directories),
        ("Ollama Connection Test", test_ollama_connection),
        ("Embedding Model Test", test_embedding_model)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your RAG chatbot is ready to use.")
        print("\n🚀 To start the application:")
        print("   streamlit run main.py")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        print("\n💡 Common solutions:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Start Ollama: ollama serve")
        print("   3. Download model: ollama pull mistral")

if __name__ == "__main__":
    main()
