# RAG Chatbot

A local AI Q&A bot that runs entirely offline using Ollama and LangChain. Upload your documents, ask questions, and get intelligent answers based on your content.

## What This Does

- **Upload documents** (PDF, TXT, MD, DOCX, CSV, images)
- **Process them automatically** into searchable chunks
- **Ask questions** and get answers based on your documents
- **Everything runs locally** - no internet needed after setup
- **OCR support** for scanned PDFs and images

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Ollama installed and running

### Installation

1. **Clone and setup**
   ```bash
   git clone <https://github.com/trying777/RAG_Bot_Internship.git>
   cd RAG
   ```

2. **Create virtual environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama**
   - Download from [ollama.ai](https://ollama.ai)
   - Start the server: `ollama serve`

5. **Download a model**
   ```bash
   ollama pull mistral
   # or
   ollama pull llama2
   #or 
   ollama pull qwen2.5:0.5b
   ```

6. **Run the app**
   ```bash
   streamlit run main.py
   ```

The app will open in your browser at `http://localhost:portnumber'

## How to Use

### 1. Upload Documents
- Drag and drop files into the upload area
- Supported: PDF, TXT, MD, DOCX, CSV, JPG, PNG, BMP, TIFF
- Click "Process & Store Documents" to ingest them

### 2. Ask Questions
- Type your question in the chat box
- The system finds relevant document chunks
- Generates an answer using your local AI model
- Shows you which documents were referenced

### 3. View Sources
- Click "View Sources" on any answer
- See exactly which documents were used
- Check similarity scores for context relevance

## Configuration

Edit `config.env` to customize:

```env
# AI Model
OLLAMA_MODEL_NAME=mistral
OLLAMA_BASE_URL=http://localhost:11434

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Search Settings
TOP_K_CHUNKS=5
TEMPERATURE=0.7
```

## Project Structure

```
RAG/
├── app/                    # Core components
│   ├── config.py          # Settings management
│   ├── document_processor.py  # File loading & chunking
│   ├── embedding_generator.py # Text embeddings
│   ├── vector_store.py    # ChromaDB storage
│   ├── ollama_client.py   # AI model client
│   └── rag_pipeline.py    # Main orchestrator
├── ui/                    # Web interface
│   └── streamlit_app.py   # Streamlit app
├── data/                  # Storage
│   ├── documents/         # Sample docs
│   ├── uploaded/          # User uploads
│   └── vector_db/         # Vector database
└── main.py                # Entry point
```

## Testing

Run these to verify everything works:

```bash
# Test basic setup
python test_setup.py

# Test OCR functionality
python test_ocr.py

# Test vector storage
python test_vector_store.py
```

## Troubleshooting

### Common Issues

**"Cannot connect to Ollama"**
- Make sure Ollama is running: `ollama serve`
- Check if port 11434 is free

**"Model not found"**
- Download the model: `ollama pull mistral`
- List available models: `ollama list`

**Import errors**
- Install dependencies: `pip install -r requirements.txt`
- Check Python version (3.8+ required)

**Memory issues with large docs**
- Reduce `CHUNK_SIZE` in config.env
- Process documents in smaller batches

### Performance Tips

- **Chunk Size**: 500-1000 for detailed answers, 1500-2000 for overviews
- **Model Choice**: Smaller models (mistral-7b) for speed, larger (llama2-70b) for quality
- **Batch Upload**: Upload multiple documents at once

## Development

### Adding New Features

1. **New document types**: Extend `DocumentProcessor` class
2. **Different embeddings**: Modify `EmbeddingGenerator` 
3. **Custom UI**: Update `streamlit_app.py`
4. **Configuration**: Add new settings to `config.env`

### Testing Changes

```bash
# Test specific component
python -c "from app.document_processor import DocumentProcessor; print('OK')"

# Run full test suite
python test_setup.py
```

## Dependencies

- **LangChain**: RAG framework
- **ChromaDB**: Vector database
- **sentence-transformers**: Text embeddings
- **Streamlit**: Web interface
- **Ollama**: Local LLM inference




