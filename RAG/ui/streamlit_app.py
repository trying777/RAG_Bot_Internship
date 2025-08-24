"""
Streamlit UI for RAG Chatbot
Provides document upload and chatbot interface
"""

import streamlit as st
import os
import time
import tempfile
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag_pipeline import RAGPipeline
from app.config import config

class StreamlitRAGApp:
    """Main Streamlit application for RAG Chatbot"""
    
    def __init__(self):
        """Initialize the Streamlit app"""
        self.setup_page()
        self.initialize_session_state()
        self.rag_pipeline = None
    
    def setup_page(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="RAG Chatbot",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Apply clean white theme CSS
        st.markdown("""
        <style>
        /* Main App Background - Clean White */
        .main {
            background-color: #FFFFFF;
            color: #262730;
        }
        .stApp {
            background-color: #FFFFFF;
            color: #262730;
        }
        
        /* Typography - Clean and readable */
        .stMarkdown {
            color: #262730;
        }
        .stHeader {
            color: #262730;
            font-weight: 600;
        }
        .stSubheader {
            color: #262730;
            font-weight: 500;
        }
        .stText {
            color: #262730;
        }
        .stMarkdown p {
            color: #262730;
            line-height: 1.6;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
            color: #262730;
            font-weight: 600;
        }
        .stMarkdown strong {
            color: #FF4B4B;
            font-weight: 600;
        }
        .stMarkdown em {
            color: #FF4B4B;
        }
        
        /* Input Fields - Clean style */
        .stTextInput > div > div > input {
            background-color: #FFFFFF;
            color: #262730;
            border: 1px solid #E0E0E0;
            border-radius: 4px;
            padding: 8px 12px;
            font-size: 16px;
        }
        .stTextInput > div > div > input:focus {
            border-color: #FF4B4B;
            box-shadow: 0 0 0 2px rgba(255, 75, 75, 0.2);
        }
        
        /* Buttons - Clean style */
        .stButton > button {
            background-color: #FF4B4B;
            color: #FFFFFF;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            font-weight: 500;
            font-size: 14px;
        }
        .stButton > button:hover {
            background-color: #E63939;
        }
        
        /* Primary Button - Special styling */
        .stButton > button[data-baseweb="button"] {
            background-color: #FF4B4B;
            color: #FFFFFF;
        }
        
        /* Progress Bars */
        .stProgress > div > div > div > div {
            background-color: #F0F0F0;
            border-radius: 2px;
        }
        .stProgress > div > div > div > div > div {
            background-color: #FF4B4B;
            border-radius: 2px;
        }
        
        /* Sidebar - Clean white */
        .stSidebar {
            background-color: #F8F9FA;
            color: #262730;
        }
        .stSidebar .sidebar-content {
            background-color: #F8F9FA;
            color: #262730;
        }
        
        /* Select Boxes */
        .stSelectbox > div > div > div > div {
            background-color: #FFFFFF;
            color: #262730;
            border: 1px solid #E0E0E0;
            border-radius: 4px;
        }
        .stSelectbox > div > div > div > div > div {
            background-color: #FFFFFF;
            color: #262730;
        }
        
        /* File Uploader - Clean style */
        .stFileUploader > div > div > div {
            background-color: #F8F9FA;
            color: #262730;
            border: 2px dashed #CCCCCC;
            border-radius: 8px;
            padding: 20px;
        }
        .stFileUploader > div > div > div:hover {
            border-color: #FF4B4B;
            background-color: #F0F0F0;
        }
        
        /* Metrics - Clean style */
        .stMetric > div > div > div {
            background-color: #F8F9FA;
            color: #262730;
            border-radius: 6px;
            padding: 12px;
            border: 1px solid #E0E0E0;
        }
        .stMetric > div > div > div > div {
            color: #FF4B4B;
            font-weight: 600;
            font-size: 14px !important;
        }
        .stMetric > div > div > div > div > div {
            font-size: 12px !important;
            color: #666666;
        }
        /* Make metric containers more compact */
        .stMetric {
            margin: 8px 0;
        }
        .stMetric > div > div > div {
            padding: 8px 10px;
            min-height: auto;
        }
        
        /* Status Messages - Clean style */
        .stSuccess {
            background-color: #D4EDDA;
            color: #155724;
            border: 1px solid #C3E6CB;
            border-radius: 4px;
            padding: 10px 12px;
        }
        .stError {
            background-color: #F8D7DA;
            color: #721C24;
            border: 1px solid #F5C6CB;
            border-radius: 4px;
            padding: 10px 12px;
        }
        .stInfo {
            background-color: #D1ECF1;
            color: #0C5460;
            border: 1px solid #BEE5EB;
            border-radius: 4px;
            padding: 10px 12px;
        }
        .stWarning {
            background-color: #FFF3CD;
            color: #856404;
            border: 1px solid #FFEEBA;
            border-radius: 4px;
            padding: 10px 12px;
        }
        
        /* Spinners */
        .stSpinner > div > div > div {
            background-color: #F8F9FA;
            color: #FF4B4B;
        }
        
        /* Expanders */
        .stExpander > div > div > div {
            background-color: #F8F9FA;
            color: #262730;
            border-radius: 4px;
            border: 1px solid #E0E0E0;
        }
        .stExpander > div > div > div > div {
            background-color: #F8F9FA;
            color: #262730;
        }
        
        /* Chat Messages - Clean style */
        .stChatMessage {
            background-color: #F8F9FA;
            color: #262730;
            border-radius: 6px;
            margin: 6px 0;
            padding: 12px;
        }
        .stChatMessage.user {
            background-color: #E3F2FD;
            color: #262730;
            border: 1px solid #BBDEFB;
        }
        .stChatMessage.assistant {
            background-color: #F8F9FA;
            color: #262730;
            border: 1px solid #E0E0E0;
        }
        
        /* Code Blocks */
        .stMarkdown code {
            background-color: #F1F3F4;
            color: #D73A49;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        }
        .stMarkdown pre {
            background-color: #F1F3F4;
            color: #D73A49;
            padding: 12px;
            border-radius: 4px;
            border: 1px solid #E0E0E0;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        }
        
        /* Links */
        .stMarkdown a {
            color: #FF4B4B;
            text-decoration: none;
        }
        .stMarkdown a:hover {
            color: #E63939;
            text-decoration: underline;
        }
        
        /* Tables */
        .stMarkdown table {
            background-color: #FFFFFF;
            color: #262730;
            border-radius: 4px;
            overflow: hidden;
        }
        .stMarkdown th {
            background-color: #F8F9FA;
            color: #262730;
            padding: 10px;
        }
        .stMarkdown td {
            background-color: #FFFFFF;
            color: #262730;
            border: 1px solid #E0E0E0;
            padding: 10px;
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #F1F1F1;
        }
        ::-webkit-scrollbar-thumb {
            background: #C1C1C1;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #A8A8A8;
        }
        
        /* Focus states */
        *:focus {
            outline: 2px solid #FF4B4B;
            outline-offset: 2px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.title("🤖 RAG Chatbot")
        st.markdown("**Local AI Q&A Bot using Ollama + LangChain**")
        st.markdown("---")
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
        
        if 'rag_initialized' not in st.session_state:
            st.session_state.rag_initialized = False
            
        if 'rag_pipeline' not in st.session_state:
            st.session_state.rag_pipeline = None
    
    def initialize_rag_pipeline(self):
        """Initialize RAG pipeline (lazy loading)"""
        if not st.session_state.rag_initialized:
            try:
                with st.spinner("🚀 Initializing..."):
                    pipeline = RAGPipeline()
                    # Store in both session state and instance
                    st.session_state.rag_pipeline = pipeline
                    self.rag_pipeline = pipeline
                    st.session_state.rag_initialized = True
                    print(f"✅ RAG Pipeline initialized: {pipeline}")
                    return True
            except Exception as e:
                st.error(f"❌ Failed to initialize RAG Pipeline: {str(e)}")
                st.error("Make sure Ollama is running: `ollama serve`")
                st.session_state.rag_initialized = False
                st.session_state.rag_pipeline = None
                return False
        else:
            # Pipeline already initialized, get from session state
            if st.session_state.rag_pipeline:
                self.rag_pipeline = st.session_state.rag_pipeline
                print(f"✅ RAG Pipeline retrieved from session: {self.rag_pipeline}")
                return True
            else:
                try:
                    # Recreate pipeline and store in session state
                    pipeline = RAGPipeline()
                    st.session_state.rag_pipeline = pipeline
                    self.rag_pipeline = pipeline
                    print(f"✅ RAG Pipeline recreated: {pipeline}")
                    return True
                except Exception as e:
                    print(f"❌ Failed to recreate RAG Pipeline: {str(e)}")
                    return False
        return True
    
    def ensure_rag_pipeline(self):
        """Ensure RAG pipeline is available, initialize if needed"""
        # First check session state
        if st.session_state.rag_pipeline:
            self.rag_pipeline = st.session_state.rag_pipeline
            return True
        
        # If not in session state, try to initialize
        if not self.rag_pipeline:
            return self.initialize_rag_pipeline()
        
        return True
    
    def render_sidebar(self):
        """Render the sidebar with controls and information"""
        with st.sidebar:
            st.header("⚙️ Controls")
            
            # Model Selection
            st.subheader("🤖 Model Selection")
            if self.rag_pipeline:
                available_models = self.rag_pipeline.get_available_models()
                current_model = self.rag_pipeline.get_current_model()
                
                # Debug info
                st.info(f"🔍 **Detected Models:** {', '.join(available_models)}")
                st.info(f"🎯 **Current Model:** {current_model}")
                
                # Model selector dropdown
                selected_model = st.selectbox(
                    "Choose LLM Model:",
                    options=available_models,
                    index=available_models.index(current_model) if current_model in available_models else 0,
                    help="Select the language model to use for generating answers"
                )
                
                # Model switching
                if selected_model != current_model:
                    if st.button(f"🔄 Switch to {selected_model}"):
                        with st.spinner(f"Switching to {selected_model}..."):
                            if self.rag_pipeline.switch_model(selected_model):
                                st.success(f"✅ Switched to {selected_model}")
                                st.rerun()
                            else:
                                st.error(f"❌ Failed to switch to {selected_model}")
                
                # Current model info
                st.info(f"**Current Model:** {current_model}")
                
                # Model management buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📥 Download Selected Model"):
                        with st.spinner(f"Downloading {selected_model}..."):
                            if self.rag_pipeline.ollama_client.pull_model(selected_model):
                                st.success(f"✅ {selected_model} downloaded successfully!")
                                st.rerun()  # Refresh to show new model
                            else:
                                st.error(f"❌ Failed to download {selected_model}")
                
                with col2:
                    if st.button("🔄 Refresh Model List"):
                        st.rerun()  # Refresh to detect new models
            
            # Document Status
            st.subheader("📊 Document Status")
            if st.session_state.uploaded_files and self.rag_pipeline:
                try:
                    vector_stats = self.rag_pipeline.vector_store.get_collection_stats()
                    total_docs = vector_stats.get('total_documents', 0)
                    
                    # Show current status
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Files", len(st.session_state.uploaded_files))
                    with col2:
                        st.metric("Chunks", total_docs)
                    with col3:
                        st.metric("Processing", "Complete" if total_docs > 0 else "Pending")
                except:
                    st.info(f"📚 {len(st.session_state.uploaded_files)} document(s) loaded")
            else:
                st.info("📤 Upload documents to get started")
            
            # Document Management
            if st.session_state.uploaded_files:
                if st.button("🗑️ Clear All Documents"):
                    # Get pipeline from session state if needed
                    pipeline = st.session_state.rag_pipeline or self.rag_pipeline
                    if pipeline and pipeline.clear_all_documents():
                        st.session_state.chat_history = []
                        st.session_state.uploaded_files = []
                        st.session_state.rag_initialized = False
                        st.session_state.rag_pipeline = None
                        self.rag_pipeline = None
                        st.rerun()
            
            # Configuration
            st.subheader("⚙️ Configuration")
            
            # Speed optimization status
            if config.CHUNK_SIZE <= 300 and config.TOP_K_CHUNKS <= 2:
                st.success("🚀 **Speed Optimized** - Fast response mode enabled")
            else:
                st.warning("🐌 **Standard Mode** - Consider reducing chunk size for speed")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"Chunk Size: {config.CHUNK_SIZE}")
                st.info(f"Chunk Overlap: {config.CHUNK_OVERLAP}")
            with col2:
                st.info(f"Top K Chunks: {config.TOP_K_CHUNKS}")
                st.info(f"Max Tokens: {config.MAX_TOKENS}")
            

    
    def render_document_upload(self):
        """Render document upload section"""
        st.header("📤 Document Upload")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['pdf', 'txt', 'md', 'docx', 'csv', 'jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, MD, DOCX, CSV, JPG, PNG, BMP, TIFF (with OCR support)"
        )
        
        # Speed Optimization Tips
        if config.CHUNK_SIZE <= 300:
            st.success("🚀 **Speed Mode**: Small chunks (300 chars) for faster processing")
        else:
            st.info("💡 **Tip**: Reduce chunk size to 300 for faster responses")
        
        # OCR Status
        if config.ENABLE_OCR:
            st.info("🔍 **OCR Enabled**: Scanned PDFs and images will be processed with text recognition")
        else:
            st.warning("⚠️ **OCR Disabled**: Only text-based documents will be processed")
        
        if uploaded_files:
            # Process uploaded files
            if st.button("🚀 Process & Store Documents"):
                if self.initialize_rag_pipeline():
                    self.process_uploaded_files(uploaded_files)
                else:
                    pass
    
    def process_uploaded_files(self, uploaded_files):
        """Process and store uploaded files"""
        # Ensure RAG pipeline is initialized
        if not self.rag_pipeline:
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Process document with detailed status
                status_text.text(f"🔄 Processing {uploaded_file.name}... (Chunking → Embedding → Storing)")
                result = self.rag_pipeline.process_and_store_document(tmp_file_path)
                
                # Clean up temp file
                os.unlink(tmp_file_path)
                
                results.append(result)
                
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Show minimal result info
                if result['success']:
                    st.success(f"✅ {uploaded_file.name}")
                else:
                    st.error(f"❌ {uploaded_file.name}")
                
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                results.append({"success": False, "error": str(e)})
        
        # Final status - Show only chunks and vector DB size
        successful = sum(1 for r in results if r["success"])
        if successful > 0:
            # Get vector database stats
            try:
                vector_stats = self.rag_pipeline.vector_store.get_collection_stats()
                total_docs = vector_stats.get('total_documents', 0)
                
                # Calculate total chunks from successful results
                total_chunks = sum(r.get('chunks_created', 0) for r in results if r['success'])
                
                # Show simple summary with all metrics
                st.success(f"📊 Document Processing Complete!")
                
                # Main metrics in 4 columns
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Number of Chunks", total_chunks)
                with col2:
                    st.metric("Vector DB Size", f"{total_docs} documents")
                with col3:
                    st.metric("Chunking Calls", len([r for r in results if r['success']]))
                with col4:
                    st.metric("Embedding Calls", total_chunks)
                
                # Additional processing details
                with st.expander("🔍 Processing Details", expanded=False):
                    st.write("**Processing Summary:**")
                    for result in results:
                        if result['success']:
                            filename = result.get('filename', 'Unknown')
                            chunks = result.get('chunks_created', 0)
                            processing_time = result.get('processing_time', 0)
                            st.write(f"✅ **{filename}**: {chunks} chunks in {processing_time}s")
                        else:
                            filename = result.get('filename', 'Unknown')
                            error = result.get('error', 'Unknown error')
                            st.write(f"❌ **{filename}**: {error}")
                
            except Exception as e:
                st.info(f"✅ {successful} document(s) processed successfully")
        
        # Update session state
        if any(r["success"] for r in results):
            # Add new files to the list
            new_files = [f.name for f in uploaded_files if f.name]
            st.session_state.uploaded_files.extend(new_files)
            
            # Ensure RAG pipeline is marked as initialized
            if not st.session_state.rag_initialized:
                st.session_state.rag_initialized = True
    
    def render_chat_interface(self):
        """Render the chat interface"""
        st.header("💬 Chat Interface")
        
        # Always show chat input if documents are processed
        if st.session_state.uploaded_files:
            # Ensure we have the pipeline from session state
            self.ensure_rag_pipeline()
            
            # Simple status
            if self.rag_pipeline:
                current_model = self.rag_pipeline.get_current_model()
                st.success(f"✅ Ready to answer questions (Model: {current_model})")
            else:
                st.warning("⚠️ RAG Pipeline not available")
            
            # Chat input
            user_question = st.text_input(
                "Ask a question about your documents:",
                placeholder="e.g., What are the main topics discussed in the documents?",
                key="user_question"
            )
            
            # Ask Question button
            if st.button("🚀 Ask Question", type="primary"):
                if user_question.strip():
                    print(f"🚀 Processing question: {user_question}")
                    self.process_question(user_question)
                else:
                    st.warning("Please enter a question")
    
    def process_question(self, question: str):
        """Process user question and generate answer"""
        if not self.rag_pipeline:
            st.error("❌ RAG Pipeline not available!")
            return
        
        # Add user question to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": question,
            "timestamp": time.time()
        })
        
        try:
            # Generate answer
            with st.spinner("🤔 Thinking..."):
                start_time = time.time()
                result = self.rag_pipeline.answer_question(question)
                processing_time = time.time() - start_time
            
            # Debug: Print result to console
            print(f"🔍 Question: {question}")
            print(f"🔍 Result: {result}")
            
            if result and result.get("success"):
                # Add bot response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "context": result.get("context", []),
                    "timestamp": time.time(),
                    "processing_time": result.get("processing_time", 0)
                })
            else:
                # Show error in chat history
                error_msg = result.get("error", "Unknown error") if result else "No response from pipeline"
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"Sorry, I couldn't answer that question. Error: {error_msg}",
                    "timestamp": time.time(),
                    "processing_time": 0
                })
                
        except Exception as e:
            print(f"❌ Error in process_question: {str(e)}")
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"Sorry, an error occurred: {str(e)}",
                "timestamp": time.time(),
                "processing_time": 0
            })
        
        # Rerun to update chat display
        st.rerun()
    
    def render_chat_history(self):
        """Render chat history with context"""
        if not st.session_state.chat_history:
            return
        
        st.header("💭 Chat History")
        
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
                    
                    # Show context if available (simplified)
                    if "context" in message and message["context"]:
                        with st.expander("📖 View Sources"):
                            for j, chunk in enumerate(message["context"]):
                                st.markdown(f"**Source:** {chunk['source']}")
                                st.markdown(f"**Content:** {chunk['content'][:150]}...")
                                st.markdown("---")
    
    def run(self):
        """Run the Streamlit application"""
        try:
            # Render sidebar first
            self.render_sidebar()
            
            # Main content area
            col1, col2 = st.columns([1, 1])
            
            with col1:
                self.render_document_upload()
            
            with col2:
                self.render_chat_interface()
            
            # Chat history
            self.render_chat_history()
            
        except Exception as e:
            st.error(f"❌ Application error: {str(e)}")
            st.exception(e)

def main():
    """Main function to run the Streamlit app"""
    app = StreamlitRAGApp()
    app.run()

if __name__ == "__main__":
    main()
