"""
Configuration module for RAG Chatbot
Loads all settings from config.env file
"""

import os
from dotenv import load_dotenv

# Load environment variables from config.env
load_dotenv('config.env')

class Config:
    """Configuration class for RAG Chatbot"""
    
    # Ollama Configuration
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_MODEL_NAME = os.getenv('OLLAMA_MODEL_NAME','qwen2.5:0.5b')
    OLLAMA_MODEL_SELECTION = os.getenv('OLLAMA_MODEL_SELECTION', 'qwen2.5:0.5b')
    
    # Available Models
    AVAILABLE_MODELS = os.getenv('AVAILABLE_MODELS', 'llama3.1:8b,mistral:latest').split(',')
    
    # Embedding Model
    EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2')
    
    # Vector Database
    VECTOR_DB_TYPE = os.getenv('VECTOR_DB_TYPE', 'chromadb')
    VECTOR_DB_PATH = os.getenv('VECTOR_DB_PATH', './data/vector_db')
    
    # Document Processing
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 300))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 50))
    MAX_CHUNKS_PER_DOCUMENT = int(os.getenv('MAX_CHUNKS_PER_DOCUMENT', 100))
    
    # RAG Configuration
    TOP_K_CHUNKS = int(os.getenv('TOP_K_CHUNKS', 2))
    TEMPERATURE = float(os.getenv('TEMPERATURE', 0.5))
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', 300))
    OLLAMA_TIMEOUT = int(os.getenv('OLLAMA_TIMEOUT', 300))
    SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', 0.3))
    
    
    # File Paths
    DOCUMENTS_PATH = os.getenv('DOCUMENTS_PATH', './data/documents')
    UPLOAD_PATH = os.getenv('UPLOAD_PATH', './data/uploaded')
    
    # OCR Configuration
    TESSERACT_CMD_PATH = os.getenv('TESSERACT_CMD_PATH', r'C:\Users\varsh\Desktop\OCR\tesseract')
    TESSDATA_PREFIX = os.getenv('TESSDATA_PREFIX', r'C:\Users\varsh\Desktop\OCR\tessdata')
    OCR_LANGUAGE = os.getenv('OCR_LANGUAGE', 'eng')
    OCR_CONFIG = os.getenv('OCR_CONFIG', '--psm 6 --oem 3')
    ENABLE_OCR = os.getenv('ENABLE_OCR', 'true').lower() == 'true'
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.DOCUMENTS_PATH,
            cls.UPLOAD_PATH,
            cls.VECTOR_DB_PATH
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Created directory: {directory}")
    
    @classmethod
    def update_model_selection(cls, new_model: str):
        """Update the selected model and save to config"""
        cls.OLLAMA_MODEL_SELECTION = new_model
        # Update the environment variable
        os.environ['OLLAMA_MODEL_SELECTION'] = new_model
        print(f"✓ Model selection updated to: {new_model}")

# Global config instance
config = Config()
