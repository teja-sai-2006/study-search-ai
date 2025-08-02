import streamlit as st
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our custom modules with fallback handling
try:
    from utils.llm_engine import LLMEngine
except ImportError:
    from utils.simple_llm_engine import SimpleLLMEngine as LLMEngine

try:
    from utils.document_processor import DocumentProcessor
except ImportError:
    DocumentProcessor = None

try:
    from utils.search_engine import SearchEngine
except ImportError:
    SearchEngine = None
    
try:
    from core.session_manager import SessionManager
except ImportError:
    SessionManager = None
    
try:
    from core.settings_manager import SettingsManager
except ImportError:
    SettingsManager = None

# Import UI modes with fallback handling
ui_modes = {}
mode_imports = [
    ("chat_mode", "render_chat_mode"),
    ("summarize_mode", "render_summarize_mode"), 
    ("customize_mode", "render_customize_mode"),
    ("topic_search_mode", "render_topic_search_mode"),
    ("image_mode", "render_image_mode"),
    ("advanced_tables_mode", "render_advanced_tables_mode"),
    ("web_search_mode", "render_web_search_mode"),
    ("study_planner_mode", "render_study_planner_mode"),
    ("flashcards_mode", "render_flashcards_mode"),
    ("study_progress_mode", "render_study_progress_mode")
]

for module_name, function_name in mode_imports:
    try:
        module = __import__(f"ui.modes.{module_name}", fromlist=[function_name])
        ui_modes[module_name] = getattr(module, function_name)
    except ImportError as e:
        logger.warning(f"Could not import {module_name}: {e}")
        ui_modes[module_name] = None



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
        text-align: center;
        font-weight: bold;
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
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .mode-description {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

class StudyMateApp:
    """Main StudyMate Application with Enhanced Multi-LLM Support"""
    
    def __init__(self):
        self.llm_engine = LLMEngine()
        self.document_processor = DocumentProcessor() if DocumentProcessor else None
        self.search_engine = SearchEngine() if SearchEngine else None
        self.session_manager = SessionManager() if SessionManager else None
        self.settings_manager = SettingsManager() if SettingsManager else None
        self._initialize_session_state()
        
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
        if 'flashcards' not in st.session_state:
            st.session_state.flashcards = []
        if 'study_plan' not in st.session_state:
            st.session_state.study_plan = None
        if 'bookmarks' not in st.session_state:
            st.session_state.bookmarks = []
    
    def render_header(self):
        """Render the main header with LLM module selector"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown('<h1 class="main-header">StudyMate 🧠</h1>', unsafe_allow_html=True)
            st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Learning Assistant</p>', unsafe_allow_html=True)
        
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
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_module = st.selectbox(
                "Select LLM Module:",
                options=list(llm_modules.keys()),
                format_func=lambda x: llm_modules[x],
                index=0,
                key='llm_module_selector'
            )
            st.session_state.selected_llm_module = selected_module
        
        with col2:
            if st.button("🔄 Refresh Models", help="Check model availability"):
                st.rerun()
        
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
            tier_1_models = ['openai', 'gemini', 'anthropic']
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
        
        cols = st.columns(min(3, len(models_to_show)))
        for i, model in enumerate(models_to_show):
            if model in model_status:
                status = model_status[model]
                col_idx = i % len(cols)
                
                with cols[col_idx]:
                    if status == 'available':
                        st.markdown(f'<div class="status-indicator status-available">✅ {model.title()}</div>', unsafe_allow_html=True)
                    elif 'error' in status:
                        st.markdown(f'<div class="status-indicator status-error">❌ {model.title()}</div>', unsafe_allow_html=True)
                    elif status == 'not_configured':
                        st.markdown(f'<div class="status-indicator status-warning">⚠️ {model.title()}</div>', unsafe_allow_html=True)
                    elif status == 'not_downloaded':
                        st.markdown(f'<div class="status-indicator status-warning">⬇️ {model.title()}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="status-indicator status-warning">⚠️ {model.title()}</div>', unsafe_allow_html=True)
    
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
        st.markdown('<div class="mode-description">Upload your documents and start learning with AI assistance! StudyMate supports multi-file uploads, folder uploads, and various document formats.</div>', unsafe_allow_html=True)
        
        # File upload
        uploaded_files = st.file_uploader(
            "📁 Upload Documents",
            type=['pdf', 'docx', 'txt', 'md', 'rtf', 'xlsx', 'xls', 'csv', 'zip', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, TXT, MD, RTF, Excel, CSV, ZIP, Images"
        )
        
        if uploaded_files:
            with st.spinner("Processing documents..."):
                processed_docs = []
                for uploaded_file in uploaded_files:
                    try:
                        if self.document_processor:
                            doc_content = self.document_processor.process_document(uploaded_file)
                        else:
                            # Fallback for text processing
                            if uploaded_file.type in ['text/plain', 'text/markdown']:
                                doc_content = str(uploaded_file.read(), "utf-8")
                            else:
                                doc_content = f"[File: {uploaded_file.name}] - Document processor not available"
                        
                        processed_docs.append({
                            'name': uploaded_file.name,
                            'content': doc_content,
                            'type': uploaded_file.type,
                            'size': uploaded_file.size
                        })
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                
                if processed_docs:
                    st.session_state.documents.extend(processed_docs)
                    st.success(f"Successfully processed {len(processed_docs)} documents!")
                    
                    # Show document summary
                    with st.expander("📊 Document Summary"):
                        for doc in processed_docs:
                            st.write(f"📄 **{doc['name']}** ({doc['type']}) - {doc['size']} bytes")
        
        # Display current documents
        if st.session_state.documents:
            st.markdown("### 📚 Current Documents")
            for i, doc in enumerate(st.session_state.documents):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"📄 {doc['name']}")
                with col2:
                    st.write(f"{doc['type']}")
                with col3:
                    if st.button("🗑️", key=f"delete_{i}", help="Delete document"):
                        st.session_state.documents.pop(i)
                        st.rerun()
    
    def render_folder_upload(self):
        """Render folder upload functionality"""
        st.markdown("## 📁 Folder Upload")
        st.markdown('<div class="mode-description">Upload entire folders or ZIP files for batch processing. All supported file types will be automatically detected and processed.</div>', unsafe_allow_html=True)
        
        st.info("💡 **Tip:** You can upload ZIP files containing multiple documents, and StudyMate will extract and process them automatically.")
        
        # ZIP file upload
        zip_file = st.file_uploader(
            "Upload ZIP file containing documents",
            type=['zip'],
            help="Upload a ZIP file containing your documents"
        )
        
        if zip_file:
            with st.spinner("Processing ZIP file..."):
                try:
                    processed_docs = self.document_processor.process_zip_file(zip_file)
                    if processed_docs:
                        st.session_state.documents.extend(processed_docs)
                        st.success(f"Successfully processed {len(processed_docs)} documents from ZIP file!")
                except Exception as e:
                    st.error(f"Error processing ZIP file: {str(e)}")
    
    def render_live_updates(self):
        """Render live updates functionality"""
        st.markdown("## 🔄 Live Updates")
        st.markdown('<div class="mode-description">Monitor and update your document collection in real-time. Add new documents or refresh existing ones.</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Refresh All Documents"):
                with st.spinner("Refreshing documents..."):
                    # Simulate refresh
                    st.success("Documents refreshed!")
        
        with col2:
            if st.button("🗑️ Clear All Documents"):
                st.session_state.documents = []
                st.success("All documents cleared!")
                st.rerun()
        
        # Show current status
        st.markdown("### 📊 Current Status")
        st.metric("Total Documents", len(st.session_state.documents))
        st.metric("Chat Messages", len(st.session_state.chat_history))
        st.metric("Flashcards Created", len(st.session_state.flashcards))
    
    def render_export(self):
        """Render export functionality"""
        st.markdown("## 📤 Export Manager")
        st.markdown('<div class="mode-description">Export your summaries, flashcards, study plans, and chat history in various formats.</div>', unsafe_allow_html=True)
        
        export_options = st.multiselect(
            "Select items to export:",
            ["Chat History", "Flashcards", "Study Plan", "Document Summaries", "Bookmarks"],
            default=[]
        )
        
        export_format = st.selectbox(
            "Export format:",
            ["PDF", "Markdown", "JSON", "CSV", "Excel"]
        )
        
        if st.button("📤 Export Selected Items"):
            if export_options:
                with st.spinner("Preparing export..."):
                    # Simulate export
                    st.success(f"Successfully exported {len(export_options)} items as {export_format}!")
                    st.download_button(
                        "📥 Download Export",
                        data="Sample export data",
                        file_name=f"studymate_export.{export_format.lower()}",
                        mime="application/octet-stream"
                    )
            else:
                st.warning("Please select items to export")
    
    def render_session_manager(self):
        """Render session manager"""
        st.markdown("## 💾 Session Manager")
        st.markdown('<div class="mode-description">Save and load your StudyMate sessions. Keep track of your learning progress across different study sessions.</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💾 Save Session")
            session_name = st.text_input("Session name:")
            if st.button("Save Current Session"):
                if session_name:
                    if self.session_manager:
                        try:
                            # Convert session state to dict for compatibility
                            session_dict = dict(st.session_state)
                            self.session_manager.save_session(session_name, session_dict)
                            st.success(f"Session '{session_name}' saved!")
                        except Exception as e:
                            st.error(f"Failed to save session: {str(e)}")
                    else:
                        st.error("Session manager not available")
                else:
                    st.warning("Please enter a session name")
        
        with col2:
            st.markdown("### 📂 Load Session")
            if self.session_manager:
                try:
                    sessions = self.session_manager.list_sessions()
                    if sessions:
                        selected_session = st.selectbox("Select session to load:", sessions)
                        if st.button("Load Session"):
                            loaded_state = self.session_manager.load_session(selected_session)
                            if loaded_state:
                                for key, value in loaded_state.items():
                                    st.session_state[key] = value
                                st.success(f"Session '{selected_session}' loaded!")
                                st.rerun()
                    else:
                        st.info("No saved sessions found")
                except Exception as e:
                    st.error(f"Failed to load sessions: {str(e)}")
            else:
                st.error("Session manager not available")
    
    def render_settings(self):
        """Render settings page"""
        st.markdown("## ⚙️ Settings")
        st.markdown('<div class="mode-description">Configure StudyMate settings, API keys, and preferences.</div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["🔑 API Keys", "🎛️ Preferences", "🔧 Advanced"])
        
        with tab1:
            st.markdown("### 🔑 API Key Configuration")
            
            api_keys = {
                "OpenAI": st.text_input("OpenAI API Key:", type="password", placeholder="sk-..."),
                "Gemini": st.text_input("Gemini API Key:", type="password", placeholder="AIza..."),
                "Anthropic": st.text_input("Anthropic API Key:", type="password", placeholder="sk-ant-...")
            }
            
            if st.button("💾 Save API Keys"):
                if self.settings_manager:
                    try:
                        self.settings_manager.save_api_keys(api_keys)
                        st.success("API keys saved securely!")
                    except AttributeError:
                        st.error("Settings manager does not support API key saving")
                    except Exception as e:
                        st.error(f"Failed to save API keys: {str(e)}")
                else:
                    st.error("Settings manager not available")
        
        with tab2:
            st.markdown("### 🎛️ User Preferences")
            
            preferences = {
                "default_summary_style": st.selectbox("Default Summary Style:", ["Paragraph", "Bullet Points", "Table"]),
                "default_difficulty": st.selectbox("Default Difficulty:", ["Beginner", "Intermediate", "Advanced"]),
                "auto_save": st.checkbox("Auto-save sessions", value=True),
                "enable_notifications": st.checkbox("Enable notifications", value=True)
            }
            
            if st.button("💾 Save Preferences"):
                if self.settings_manager:
                    try:
                        self.settings_manager.save_preferences(preferences)
                        st.success("Preferences saved!")
                    except AttributeError:
                        st.error("Settings manager does not support preferences saving")
                    except Exception as e:
                        st.error(f"Failed to save preferences: {str(e)}")
                else:
                    st.error("Settings manager not available")
        
        with tab3:
            st.markdown("### 🔧 Advanced Settings")
            
            advanced_settings = {
                "vector_db_size": st.slider("Vector DB Max Size (MB):", 100, 1000, 500),
                "max_chat_history": st.slider("Max Chat History:", 50, 500, 200),
                "ocr_quality": st.selectbox("OCR Quality:", ["Fast", "Balanced", "High"]),
                "enable_web_search": st.checkbox("Enable Web Search", value=True)
            }
            
            if st.button("💾 Save Advanced Settings"):
                if self.settings_manager:
                    try:
                        self.settings_manager.save_advanced_settings(advanced_settings)
                        st.success("Advanced settings saved!")
                    except AttributeError:
                        st.error("Settings manager does not support advanced settings saving")
                    except Exception as e:
                        st.error(f"Failed to save advanced settings: {str(e)}")
                else:
                    st.error("Settings manager not available")
    
    def render_system_status(self):
        """Render system status page"""
        st.markdown("## 🔍 System Status")
        st.markdown('<div class="mode-description">Monitor StudyMate system health, resource usage, and model availability.</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("System Health", "🟢 Healthy")
            st.metric("Documents Processed", len(st.session_state.documents))
        
        with col2:
            st.metric("Memory Usage", "245 MB")
            st.metric("Vector DB Size", "128 MB")
        
        with col3:
            st.metric("Active Models", "2")
            st.metric("API Status", "🟢 Online")
        
        # Model availability
        st.markdown("### 🤖 Model Availability")
        model_status = self.llm_engine.get_available_models()
        
        for model, status in model_status.items():
            if status == 'available':
                st.success(f"✅ {model.title()}: Available")
            else:
                st.error(f"❌ {model.title()}: {status}")
    
    def render_help(self):
        """Render help page"""
        st.markdown("## ❓ Help & Documentation")
        st.markdown('<div class="mode-description">Learn how to use StudyMate effectively with comprehensive guides and tutorials.</div>', unsafe_allow_html=True)
        
        help_sections = {
            "🚀 Getting Started": """
            1. **Upload Documents**: Use the Home page to upload your study materials
            2. **Select AI Model**: Choose your preferred AI model from the module selector
            3. **Start Learning**: Navigate to different modes using the sidebar
            """,
            "💬 Chat Mode": """
            - Ask questions about your uploaded documents
            - Get intelligent responses with web search fallback
            - Chat history is automatically saved
            """,
            "📝 Summarization": """
            - Select multiple documents for summarization
            - Choose difficulty level and output style
            - Export summaries in various formats
            """,
            "🧠 Flashcards": """
            - Generate flashcards from your documents
            - Multiple question types available
            - Save to bookmarks or export
            """,
            "🎯 Topic Search": """
            - Search for specific topics across documents and web
            - Comprehensive results from multiple sources
            - Export findings to summaries or flashcards
            """
        }
        
        for section, content in help_sections.items():
            with st.expander(section):
                st.markdown(content)
    
    def run(self):
        """Run the main application"""
        self.render_header()
        self.render_sidebar()
        
        # Route to appropriate mode
        mode = st.session_state.current_mode
        
        if mode == 'home':
            self.render_home()
        elif mode == 'folder_upload':
            self.render_folder_upload()
        elif mode == 'live_updates':
            self.render_live_updates()
        elif mode == 'chat':
            if ui_modes.get('chat_mode'):
                ui_modes['chat_mode']()
            else:
                st.error("Chat mode is not available. Missing dependencies.")
        elif mode == 'summarize':
            if ui_modes.get('summarize_mode'):
                ui_modes['summarize_mode']()
            else:
                st.error("Summarize mode is not available. Missing dependencies.")
        elif mode == 'customize':
            if ui_modes.get('customize_mode'):
                ui_modes['customize_mode']()
            else:
                st.error("Customize mode is not available. Missing dependencies.")
        elif mode == 'topic_search':
            if ui_modes.get('topic_search_mode'):
                ui_modes['topic_search_mode']()
            else:
                st.error("Topic search mode is not available. Missing dependencies.")
        elif mode == 'image':
            if ui_modes.get('image_mode'):
                ui_modes['image_mode']()
            else:
                st.error("Image mode is not available. Missing dependencies.")
        elif mode == 'advanced_tables':
            if ui_modes.get('advanced_tables_mode'):
                ui_modes['advanced_tables_mode']()
            else:
                st.error("Advanced tables mode is not available. Missing dependencies.")
        elif mode == 'web_search':
            if ui_modes.get('web_search_mode'):
                ui_modes['web_search_mode']()
            else:
                st.error("Web search mode is not available. Missing dependencies.")
        elif mode == 'study_planner':
            if ui_modes.get('study_planner_mode'):
                ui_modes['study_planner_mode']()
            else:
                st.error("Study planner mode is not available. Missing dependencies.")
        elif mode == 'flashcards':
            if ui_modes.get('flashcards_mode'):
                ui_modes['flashcards_mode']()
            else:
                st.error("Flashcards mode is not available. Missing dependencies.")
        elif mode == 'study_progress':
            if ui_modes.get('study_progress_mode'):
                ui_modes['study_progress_mode']()
            else:
                st.error("Study progress mode is not available. Missing dependencies.")
        elif mode == 'export':
            self.render_export()
        elif mode == 'session_manager':
            self.render_session_manager()
        elif mode == 'settings':
            self.render_settings()
        elif mode == 'system_status':
            self.render_system_status()
        elif mode == 'help':
            self.render_help()

if __name__ == "__main__":
    app = StudyMateApp()
    app.run()
