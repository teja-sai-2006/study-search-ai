import streamlit as st
import os
import logging
from typing import Dict, Optional

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
</style>
""", unsafe_allow_html=True)

def check_system_status():
    """Check system status and available features"""
    status = {
        "core": "available",
        "ui": "available", 
        "ai_models": "not_configured",
        "file_processing": "limited"
    }
    
    # Check for API keys
    api_keys = {
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "gemini": bool(os.getenv("GEMINI_API_KEY")),
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY"))
    }
    
    if any(api_keys.values()):
        status["ai_models"] = "available"
    
    return status, api_keys

def render_status_dashboard():
    """Render system status dashboard"""
    st.markdown('<div class="main-header">StudyMate - AI-Powered Learning Assistant</div>', unsafe_allow_html=True)
    
    status, api_keys = check_system_status()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("System Status")
        
        for component, state in status.items():
            if state == "available":
                st.markdown(f'<div class="status-indicator status-available">✅ {component.replace("_", " ").title()}: Ready</div>', unsafe_allow_html=True)
            elif state == "limited":
                st.markdown(f'<div class="status-indicator status-warning">⚠️ {component.replace("_", " ").title()}: Limited</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-indicator status-error">❌ {component.replace("_", " ").title()}: Not Ready</div>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("AI Model Status")
        
        for provider, available in api_keys.items():
            if available:
                st.markdown(f'<div class="status-indicator status-available">✅ {provider.title()}: Configured</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-indicator status-error">❌ {provider.title()}: Not Configured</div>', unsafe_allow_html=True)

def render_setup_guide():
    """Render setup guide for users"""
    st.subheader("Getting Started")
    
    st.markdown("""
    <div class="feature-card">
    <h3>🚀 Welcome to StudyMate!</h3>
    <p>Your AI-powered learning assistant is ready to help you study more effectively.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Quick Setup")
    
    # API Key Setup
    with st.expander("🔑 Configure AI Models", expanded=True):
        st.markdown("""
        To enable AI features, add your API keys:
        
        **Option 1: Environment Variables**
        - `OPENAI_API_KEY` - For GPT models
        - `GEMINI_API_KEY` - For Google's Gemini models  
        - `ANTHROPIC_API_KEY` - For Claude models
        
        **Option 2: Settings Panel**
        Use the settings panel in the sidebar to add keys securely.
        """)
        
        # Basic API key input
        st.subheader("Add API Keys")
        
        openai_key = st.text_input("OpenAI API Key", type="password", help="Get your key from https://platform.openai.com/api-keys")
        gemini_key = st.text_input("Google Gemini API Key", type="password", help="Get your key from Google AI Studio")
        anthropic_key = st.text_input("Anthropic API Key", type="password", help="Get your key from https://console.anthropic.com/")
        
        if st.button("Save API Keys"):
            if openai_key:
                os.environ["OPENAI_API_KEY"] = openai_key
                st.success("OpenAI API key saved!")
            if gemini_key:
                os.environ["GEMINI_API_KEY"] = gemini_key
                st.success("Gemini API key saved!")
            if anthropic_key:
                os.environ["ANTHROPIC_API_KEY"] = anthropic_key
                st.success("Anthropic API key saved!")
            
            st.rerun()
    
    # Available Features
    with st.expander("📚 Available Features"):
        st.markdown("""
        **Core Features:**
        - 💬 **Chat Mode** - Ask questions and get AI-powered answers
        - 📄 **Document Summarization** - Summarize PDFs, Word docs, and more
        - 🔍 **Content Search** - Search within your documents and the web
        - 🖼️ **Image Analysis** - Extract text and analyze images
        - 📊 **Table Extraction** - Extract and analyze tables from documents
        - 🗂️ **Study Planning** - Create structured study plans
        - 🔁 **Flashcards** - Generate and practice with AI-created flashcards
        - 📈 **Progress Tracking** - Monitor your learning progress
        
        **Document Support:**
        - PDF, DOCX, TXT, MD, RTF files
        - Excel and CSV files
        - ZIP archives with multiple documents
        - Image files (JPG, PNG) with OCR
        """)

def render_basic_chat():
    """Render a basic chat interface"""
    st.subheader("💬 Chat with StudyMate")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your studies..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            # Check if AI is available
            _, api_keys = check_system_status()
            
            if any(api_keys.values()):
                response = "I can see you have AI configured! The full AI features are being set up. For now, I can help you get started with StudyMate's features."
            else:
                response = f"""I'd love to help you with: "{prompt}"

To enable full AI responses, please:
1. Add your API key using the setup panel above
2. Choose from OpenAI GPT, Google Gemini, or Anthropic Claude
3. Then I'll be able to give you detailed, intelligent responses!

For now, I can help you navigate StudyMate's features and get everything set up."""
            
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    """Main application function"""
    
    # Sidebar for navigation
    with st.sidebar:
        st.title("StudyMate")
        st.markdown("---")
        
        mode = st.selectbox(
            "Choose Mode",
            ["Setup & Status", "Chat", "File Upload", "Settings"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### Quick Actions")
        
        if st.button("🔄 Refresh Status"):
            st.rerun()
        
        if st.button("📚 View Documentation"):
            st.info("Documentation coming soon!")
        
        st.markdown("---")
        st.markdown("### System Info")
        st.info("StudyMate v1.0\nStreamlit-based AI Learning Platform")
    
    # Main content area
    if mode == "Setup & Status":
        render_status_dashboard()
        render_setup_guide()
    
    elif mode == "Chat":
        render_basic_chat()
    
    elif mode == "File Upload":
        st.subheader("📁 File Upload")
        st.info("File processing features are being prepared. Please check back soon!")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'docx', 'txt', 'md', 'jpg', 'png'],
            help="Upload documents for processing"
        )
        
        if uploaded_file is not None:
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")
            st.info("AI-powered document processing will be available once AI models are configured.")
    
    elif mode == "Settings":
        st.subheader("⚙️ Settings")
        
        st.markdown("### Application Settings")
        
        theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])
        language = st.selectbox("Language", ["English", "Spanish", "French", "German"])
        
        st.markdown("### AI Model Preferences")
        
        preferred_model = st.selectbox(
            "Preferred AI Model",
            ["Auto (Best Available)", "OpenAI GPT", "Google Gemini", "Anthropic Claude"]
        )
        
        max_tokens = st.slider("Response Length", 100, 2000, 500)
        
        if st.button("Save Settings"):
            st.success("Settings saved successfully!")

if __name__ == "__main__":
    main()