"""
Main Entry Point for RAG Chatbot
Run this file to start the Streamlit application
"""

import os
import sys
from pathlib import Path

# Add the app directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def main():
    """Main function to run the RAG chatbot"""
    print("🚀 Starting RAG Chatbot...")
    
    # Check if required directories exist
    data_dirs = [
        "data/documents",
        "data/uploaded", 
        "data/vector_db"
    ]
    
    for dir_path in data_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Directory ready: {dir_path}")
    
    # Import and run Streamlit app
    try:
        from ui.streamlit_app import main as run_streamlit
        run_streamlit()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error running app: {e}")

if __name__ == "__main__":
    main()
