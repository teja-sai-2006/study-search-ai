import streamlit as st
from typing import List, Dict, Any
import time

def render_chat_interface(messages: List[Dict[str, str]], on_send_message=None):
    """Render chat interface with message history"""
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display messages
        for i, message in enumerate(messages):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
    
    # Input area
    user_input = st.chat_input("Ask a question about your documents...")
    
    if user_input and on_send_message:
        on_send_message(user_input)
        st.rerun()

def render_streaming_response(response_placeholder, response_text: str):
    """Render streaming response effect"""
    words = response_text.split()
    displayed_text = ""
    
    for word in words:
        displayed_text += word + " "
        response_placeholder.markdown(displayed_text)
        time.sleep(0.05)  # Streaming effect

def render_message_with_sources(content: str, sources: List[Dict[str, Any]] = None):
    """Render message with source citations"""
    st.markdown(content)
    
    if sources:
        with st.expander("📚 Sources"):
            for i, source in enumerate(sources):
                st.markdown(f"**{i+1}.** {source.get('title', 'Unknown')}")
                if source.get('url'):
                    st.markdown(f"🔗 [Link]({source['url']})")
                if source.get('score'):
                    st.markdown(f"Relevance: {source['score']:.2f}")
                st.markdown("---")

def render_chat_controls():
    """Render chat control buttons"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        if st.button("💾 Save Chat"):
            # Implementation for saving chat
            st.success("Chat saved!")
    
    with col3:
        if st.button("📤 Export Chat"):
            # Implementation for exporting chat
            st.success("Chat exported!")
