import streamlit as st
import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional
import tempfile
import time

# Import our custom modules
from utils.llm_engine import LLMEngine
from utils.document_processor import DocumentProcessor
from utils.search_engine import SearchEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="StudyMate - AI-Powered Learning Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .module-selector {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .status-indicator {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .status-available { background-color: #d4edda; color: #155724; }
    .status-error { background-color: #f8d7da; color: #721c24; }
    .status-warning { background-color: #fff3cd; color: #856404; }
    .feature-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StudyMateApp:
    """Main StudyMate Application with Enhanced Multi-LLM Support"""
    
    def __init__(self):
        self.llm_engine = LLMEngine()
        self.document_processor = DocumentProcessor()
        self.search_engine = SearchEngine()
        self.session_state = self._initialize_session_state()
        
    def _initialize_session_state(self):
        """Initialize session state variables"""
        if 'documents' not in st.session_state:
            st.session_state.documents = []
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'current_mode' not in st.session_state:
            st.session_state.current_mode = 'home'
        if 'selected_llm_module' not in st.session_state:
            st.session_state.selected_llm_module = 'auto_smart'
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = None
        if 'search_results' not in st.session_state:
            st.session_state.search_results = []
        if 'processed_content' not in st.session_state:
            st.session_state.processed_content = ""
        if 'table_data' not in st.session_state:
            st.session_state.table_data = []
        return st.session_state
    
    def render_header(self):
        """Render the main header with LLM module selector"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.markdown("🧠")
        
        with col2:
            st.markdown('<h1 class="main-header">StudyMate</h1>', unsafe_allow_html=True)
            st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Learning Assistant</p>', unsafe_allow_html=True)
        
        with col3:
            st.markdown("🧠")
        
        # LLM Module Selector
        self.render_llm_module_selector()
    
    def render_llm_module_selector(self):
        """Render the enhanced LLM module selector"""
        st.markdown('<div class="module-selector">', unsafe_allow_html=True)
        st.markdown("### 🎛️ LLM Module Selector")
        
        # Load AI config
        try:
            with open("config/ai_config.json", 'r') as f:
                ai_config = json.load(f)
            llm_modules_config = ai_config.get('ui_config', {}).get('llm_modules', {})
        except FileNotFoundError:
            llm_modules_config = {}
        
        # Module selection
        llm_modules = {}
        for key, config in llm_modules_config.items():
            llm_modules[key] = config.get('name', key)
        
        # Fallback if config not found
        if not llm_modules:
            llm_modules = {
                'auto_smart': '🤖 Auto-Smart (Recommended)',
                'premium_apis': '🔥 Premium APIs',
                'offline_custom': '💽 Offline Local Models'
            }
        
        selected_module = st.selectbox(
            "Select LLM Module:",
            options=list(llm_modules.keys()),
            format_func=lambda x: llm_modules[x],
            index=0,
            key='llm_module_selector'
        )
        
        st.session_state.selected_llm_module = selected_module
        
        # Show offline model selector if offline mode selected
        if selected_module == 'offline_custom':
            self.render_offline_model_selector()
        
        # Model status display
        self.render_model_status()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_model_status(self):
        """Render model status indicators"""
        model_status = self.llm_engine.get_available_models()
        
        st.markdown("#### 📊 Model Status")
        
        # Load AI config for model grouping
        try:
            with open("config/ai_config.json", 'r') as f:
                ai_config = json.load(f)
            tier_1_models = list(ai_config.get('llm_config', {}).get('tier_1_online_models', {}).keys())
            offline_models = list(ai_config.get('llm_config', {}).get('offline_models', {}).keys())
        except FileNotFoundError:
            tier_1_models = ['openai', 'gemini', 'anthropic', 'ibm_watson']
            offline_models = ['dialo_gpt_small', 'flan_t5_base', 'granite_3_3_2b', 'mistral_7b_instruct', 'phi_2']
        
        # Group models by module
        module_models = {
            'premium_apis': tier_1_models,
            'offline_custom': offline_models,
            'auto_smart': tier_1_models + offline_models
        }
        
        current_module = st.session_state.selected_llm_module
        if current_module in module_models:
            models_to_show = module_models[current_module]
        else:
            models_to_show = list(model_status.keys())
        
        cols = st.columns(3)
        for i, model in enumerate(models_to_show):
            if model in model_status:
                status = model_status[model]
                col_idx = i % 3
                
                with cols[col_idx]:
                    if status == 'available':
                        st.markdown(f'<div class="status-indicator status-available">✅ {model}</div>', unsafe_allow_html=True)
                    elif 'error' in status:
                        st.markdown(f'<div class="status-indicator status-error">❌ {model}</div>', unsafe_allow_html=True)
                    elif status == 'not_configured':
                        st.markdown(f'<div class="status-indicator status-warning">⚠️ {model}</div>', unsafe_allow_html=True)
                    elif status == 'not_downloaded':
                        st.markdown(f'<div class="status-indicator status-warning">⬇️ {model}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="status-indicator status-warning">⚠️ {model}</div>', unsafe_allow_html=True)
    
    def render_offline_model_selector(self):
        """Render offline model selector"""
        st.markdown("#### 💽 Select Offline Model")
        
        try:
            with open("config/ai_config.json", 'r') as f:
                ai_config = json.load(f)
            offline_models = ai_config.get('llm_config', {}).get('offline_models', {})
        except FileNotFoundError:
            offline_models = {}
        
        if offline_models:
            model_options = {}
            for key, config in offline_models.items():
                model_options[key] = f"{config.get('name', key)} ({config.get('size_mb', 0)}MB)"
            
            selected_offline_model = st.selectbox(
                "Choose offline model (only one active at a time):",
                options=list(model_options.keys()),
                format_func=lambda x: model_options[x],
                key='offline_model_selector'
            )
            
            if selected_offline_model:
                st.session_state.selected_offline_model = selected_offline_model
                st.info(f"✅ Selected: {offline_models[selected_offline_model].get('name', selected_offline_model)}")
        else:
            st.warning("No offline models configured")
    
    def render_sidebar(self):
        """Render the complete sidebar navigation"""
        st.sidebar.markdown("## 📂 Navigation")
        
        # Document Management
        st.sidebar.markdown("### 📂 Document Management")
        if st.sidebar.button("🏠 Home", key="nav_home"):
            st.session_state.current_mode = 'home'
        
        if st.sidebar.button("📁 Folder Upload", key="nav_folder"):
            st.session_state.current_mode = 'folder_upload'
        
        if st.sidebar.button("🔄 Live Updates", key="nav_live"):
            st.session_state.current_mode = 'live_updates'
        
        # AI Interaction Modes
        st.sidebar.markdown("### 🤖 AI Interaction")
        if st.sidebar.button("💬 Chat Mode", key="nav_chat"):
            st.session_state.current_mode = 'chat'
        
        if st.sidebar.button("📝 Summarize Mode", key="nav_summarize"):
            st.session_state.current_mode = 'summarize'
        
        if st.sidebar.button("🛠️ Customize Mode", key="nav_customize"):
            st.session_state.current_mode = 'customize'
        
        if st.sidebar.button("🎯 Topic Search", key="nav_topic_search"):
            st.session_state.current_mode = 'topic_search'
        
        # Advanced Analysis
        st.sidebar.markdown("### 📊 Advanced Analysis")
        if st.sidebar.button("🖼️ Image Mode", key="nav_image"):
            st.session_state.current_mode = 'image'
        
        if st.sidebar.button("📊 Advanced Tables", key="nav_tables"):
            st.session_state.current_mode = 'advanced_tables'
        
        if st.sidebar.button("🌐 Web Search", key="nav_web"):
            st.session_state.current_mode = 'web_search'
        
        # Study Tools
        st.sidebar.markdown("### 📚 Study Tools")
        if st.sidebar.button("📚 Study Planner", key="nav_planner"):
            st.session_state.current_mode = 'study_planner'
        
        if st.sidebar.button("🧠 Flashcards/Notes", key="nav_flashcards"):
            st.session_state.current_mode = 'flashcards'
        
        if st.sidebar.button("📈 Study Progress", key="nav_progress"):
            st.session_state.current_mode = 'study_progress'
        
        # Export & Productivity
        st.sidebar.markdown("### 📤 Export & Productivity")
        if st.sidebar.button("📤 Export", key="nav_export"):
            st.session_state.current_mode = 'export'
        
        if st.sidebar.button("💾 Session Manager", key="nav_session"):
            st.session_state.current_mode = 'session_manager'
        
        # System & Configuration
        st.sidebar.markdown("### ⚙️ System & Configuration")
        if st.sidebar.button("⚙️ Settings", key="nav_settings"):
            st.session_state.current_mode = 'settings'
        
        if st.sidebar.button("🔍 System Status", key="nav_status"):
            st.session_state.current_mode = 'system_status'
        
        if st.sidebar.button("❓ Help", key="nav_help"):
            st.session_state.current_mode = 'help'
        
        # Session info
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Current Mode:** {st.session_state.current_mode.replace('_', ' ').title()}")
        st.sidebar.markdown(f"**Documents:** {len(st.session_state.documents)}")
        st.sidebar.markdown(f"**Chat History:** {len(st.session_state.chat_history)}")
    
    def render_home(self):
        """Render the home page with document upload"""
        st.markdown("## 🏠 Welcome to StudyMate")
        st.markdown("Upload your documents and start learning with AI assistance!")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'docx', 'txt', 'md', 'rtf', 'xlsx', 'xls', 'csv', 'zip', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, TXT, MD, RTF, Excel, CSV, ZIP, Images"
        )
        
        if uploaded_files:
            with st.spinner("Processing documents..."):
                for uploaded_file in uploaded_files:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    # Process document
                    result = self.document_processor.process_document(tmp_file_path)
                    
                    if 'error' not in result:
                        st.session_state.documents.append({
                            'name': uploaded_file.name,
                            'content': result.get('content', ''),
                            'tables': result.get('tables', []),
                            'metadata': result.get('metadata', {}),
                            'type': result.get('type', 'unknown')
                        })
                        
                        st.success(f"✅ Processed: {uploaded_file.name}")
                        
                        # Index for search
                        if st.session_state.documents:
                            self.search_engine.index_documents(st.session_state.documents)
                    else:
                        st.error(f"❌ Error processing {uploaded_file.name}: {result['error']}")
                    
                    # Clean up temp file
                    os.unlink(tmp_file_path)
        
        # Display processed documents
        if st.session_state.documents:
            st.markdown("### 📄 Processed Documents")
            for i, doc in enumerate(st.session_state.documents):
                with st.expander(f"📄 {doc['name']} ({doc['type']})"):
                    st.markdown(f"**Type:** {doc['type']}")
                    st.markdown(f"**Content Length:** {len(doc['content'])} characters")
                    if doc['tables']:
                        st.markdown(f"**Tables Found:** {len(doc['tables'])}")
                    if doc['metadata']:
                        st.markdown("**Metadata:**")
                        st.json(doc['metadata'])
    
    def render_chat_mode(self):
        """Render the chat mode interface"""
        st.markdown("## 💬 Chat Mode")
        st.markdown("Ask questions about your uploaded documents!")
        
        # Chat interface
        user_input = st.text_input("Ask a question:", key="chat_input")
        
        if st.button("Send", key="chat_send") and user_input:
            with st.spinner("Generating response..."):
                # Get document context
                context = ""
                if st.session_state.documents:
                    context = "\n\n".join([doc['content'] for doc in st.session_state.documents])
                
                # Generate response
                response, model_used, confidence = asyncio.run(
                    self.llm_engine.generate_response(
                        prompt=user_input,
                        context=context,
                        model_name=st.session_state.selected_model
                    )
                )
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'user': user_input,
                    'assistant': response,
                    'model': model_used,
                    'confidence': confidence,
                    'timestamp': time.time()
                })
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("### 💬 Chat History")
            for i, chat in enumerate(st.session_state.chat_history):
                with st.expander(f"💬 Chat {i+1} (Model: {chat['model']}, Confidence: {chat['confidence']:.2f})"):
                    st.markdown(f"**You:** {chat['user']}")
                    st.markdown(f"**Assistant:** {chat['assistant']}")
    
    def render_topic_search(self):
        """Render the topic search interface"""
        st.markdown("## 🎯 Topic Search")
        st.markdown("Search for concepts and topics across your documents and the web!")
        
        # Search options
        col1, col2 = st.columns(2)
        with col1:
            include_web = st.checkbox("Include Web Search", value=True)
        with col2:
            include_wikipedia = st.checkbox("Include Wikipedia", value=True)
        
        # Search input
        search_query = st.text_input("Search for a topic:", key="topic_search_input")
        
        if st.button("Search", key="topic_search_button") and search_query:
            with st.spinner("Searching..."):
                # Perform hybrid search
                results = asyncio.run(
                    self.search_engine.hybrid_search(
                        query=search_query,
                        include_web=include_web,
                        include_wikipedia=include_wikipedia
                    )
                )
                
                st.session_state.search_results = results
        
        # Display search results
        if st.session_state.search_results and 'combined_results' in st.session_state.search_results:
            st.markdown("### 🔍 Search Results")
            
            for i, result in enumerate(st.session_state.search_results['combined_results']):
                with st.expander(f"📄 Result {i+1} - {result.get('source_type', 'unknown')} (Confidence: {result.get('confidence', 0):.2f})"):
                    if 'title' in result:
                        st.markdown(f"**Title:** {result['title']}")
                    st.markdown(f"**Content:** {result.get('content', '')[:500]}...")
                    if 'url' in result:
                        st.markdown(f"**Source:** {result['url']}")
    
    def render_settings(self):
        """Render the settings page"""
        st.markdown("## ⚙️ Settings")
        
        # API Key Management
        st.markdown("### 🔑 API Key Management")
        
        # Load current config
        try:
            with open("config/api_keys.json", 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            config = {}
        
        # OpenAI
        openai_key = st.text_input("OpenAI API Key:", value=config.get('openai', {}).get('api_key', ''), type="password")
        if openai_key:
            config.setdefault('openai', {})['api_key'] = openai_key
        
        # Gemini
        gemini_key = st.text_input("Google Gemini API Key:", value=config.get('gemini', {}).get('api_key', ''), type="password")
        if gemini_key:
            config.setdefault('gemini', {})['api_key'] = gemini_key
        
        # Anthropic
        anthropic_key = st.text_input("Anthropic API Key:", value=config.get('anthropic', {}).get('api_key', ''), type="password")
        if anthropic_key:
            config.setdefault('anthropic', {})['api_key'] = anthropic_key
        
        # IBM Watson
        ibm_key = st.text_input("IBM Watson API Key:", value=config.get('ibm_watson', {}).get('api_key', ''), type="password")
        if ibm_key:
            config.setdefault('ibm_watson', {})['api_key'] = ibm_key
        
        ibm_project = st.text_input("IBM Watson Project ID:", value=config.get('ibm_watson', {}).get('project_id', ''))
        if ibm_project:
            config.setdefault('ibm_watson', {})['project_id'] = ibm_project
        
        # HuggingFace
        hf_key = st.text_input("HuggingFace Token (Optional):", value=config.get('huggingface', {}).get('api_key', ''), type="password")
        if hf_key:
            config.setdefault('huggingface', {})['api_key'] = hf_key
        
        # Save configuration
        if st.button("Save Configuration"):
            os.makedirs("config", exist_ok=True)
            with open("config/api_keys.json", 'w') as f:
                json.dump(config, f, indent=2)
            st.success("✅ Configuration saved!")
            
            # Reinitialize LLM engine
            self.llm_engine = LLMEngine()
        
        # Privacy Controls
        st.markdown("### 🔒 Privacy Controls")
        web_search_enabled = st.checkbox("Enable Web Search", value=True)
        data_collection = st.checkbox("Allow Data Collection for Improvement", value=False)
        
        # Performance Settings
        st.markdown("### ⚡ Performance Settings")
        max_tokens = st.slider("Max Tokens:", min_value=100, max_value=4000, value=1000)
        temperature = st.slider("Temperature:", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
        
        # Model Status
        st.markdown("### 📊 Model Status")
        model_status = self.llm_engine.get_model_status_summary()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Models", model_status['total_models'])
        with col2:
            st.metric("Available", model_status['available_models'])
        with col3:
            st.metric("Failed", model_status['failed_models'])
        with col4:
            st.metric("Not Configured", model_status['not_configured'])
    
    def render_system_status(self):
        """Render the system status page"""
        st.markdown("## 🔍 System Status")
        
        # LLM Engine Status
        st.markdown("### 🤖 LLM Engine Status")
        model_status = self.llm_engine.get_available_models()
        
        for model, status in model_status.items():
            if status == 'available':
                st.success(f"✅ {model}")
            elif 'error' in status:
                st.error(f"❌ {model}: {status}")
            elif status == 'not_configured':
                st.warning(f"⚠️ {model}: Not configured")
            else:
                st.info(f"⬇️ {model}: {status}")
        
        # Search Engine Status
        st.markdown("### 🔍 Search Engine Status")
        search_stats = self.search_engine.get_index_stats()
        
        st.metric("Document Chunks", search_stats['total_chunks'])
        st.metric("Index Available", "Yes" if search_stats['index_available'] else "No")
        st.metric("Embedding Models", len(search_stats['embedding_models']))
        
        # Document Processor Status
        st.markdown("### 📄 Document Processor Status")
        st.metric("Processed Documents", len(st.session_state.documents))
        st.metric("Total Tables", sum(len(doc.get('tables', [])) for doc in st.session_state.documents))
    
    def render_help(self):
        """Render the help page"""
        st.markdown("## ❓ Help & Documentation")
        
        st.markdown("### 🚀 Getting Started")
        st.markdown("""
        1. **Upload Documents**: Start by uploading your study materials (PDF, DOCX, TXT, etc.)
        2. **Choose LLM Module**: Select the AI model category that suits your needs
        3. **Start Learning**: Use Chat Mode, Topic Search, or other features to interact with your documents
        """)
        
        st.markdown("### 🎛️ LLM Modules")
        st.markdown("""
        - **🤖 Auto-Smart**: Automatically selects the best available model
        - **🔥 Premium APIs**: Uses OpenAI, Gemini, Claude, and IBM Watson (requires API keys)
        - **💎 IBM Granite**: IBM's specialized models for enterprise tasks
        - **🤗 HuggingFace**: Open-source models from the community
        - **🖥️ Offline Only**: Local models that work without internet
        - **🆓 Free Models**: Models that don't require API keys
        """)
        
        st.markdown("### 📱 Features")
        st.markdown("""
        - **💬 Chat Mode**: Conversational Q&A with your documents
        - **📝 Summarize Mode**: Generate summaries of your documents
        - **🎯 Topic Search**: Search for concepts across documents and web
        - **🖼️ Image Mode**: Analyze images and diagrams
        - **📊 Advanced Tables**: Extract and analyze table data
        - **📚 Study Tools**: Flashcards, study planner, and progress tracking
        """)
        
        st.markdown("### 🔧 Troubleshooting")
        st.markdown("""
        - **No API Keys**: Use Free Models or Offline Only modules
        - **Slow Performance**: Try smaller models or reduce max tokens
        - **Document Processing Errors**: Check file format and size
        - **Search Not Working**: Ensure documents are properly uploaded and indexed
        """)
    
    def render_mode_content(self):
        """Render content based on current mode"""
        mode = st.session_state.current_mode
        
        if mode == 'home':
            self.render_home()
        elif mode == 'chat':
            self.render_chat_mode()
        elif mode == 'topic_search':
            self.render_topic_search()
        elif mode == 'settings':
            self.render_settings()
        elif mode == 'system_status':
            self.render_system_status()
        elif mode == 'help':
            self.render_help()
        else:
            # Placeholder for other modes
            st.markdown(f"## {mode.replace('_', ' ').title()}")
            st.info(f"This feature is coming soon! Mode: {mode}")
    
    def run(self):
        """Main application runner"""
        # Render header with LLM module selector
        self.render_header()
        
        # Render sidebar navigation
        self.render_sidebar()
        
        # Render main content based on current mode
        self.render_mode_content()

def main():
    """Main function to run the StudyMate application"""
    app = StudyMateApp()
    app.run()

if __name__ == "__main__":
    main() 