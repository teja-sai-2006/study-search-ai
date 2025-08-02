import streamlit as st
from typing import List, Dict, Any
import logging
from utils.llm_engine import LLMEngine
from utils.search_engine import SearchEngine
from ui.components.chat_interface import render_chat_interface, render_chat_controls

logger = logging.getLogger(__name__)

def render_chat_mode():
    """Render chat mode with document Q&A and web search fallback"""
    st.markdown("## 💬 Chat Mode")
    st.markdown('<div class="mode-description">Ask questions about your documents or get information from the web. The AI will search your uploaded documents first, then fall back to web sources if needed.</div>', unsafe_allow_html=True)
    
    # Initialize components
    llm_engine = LLMEngine()
    search_engine = SearchEngine()
    
    # Check if documents are available
    if not st.session_state.documents:
        st.warning("📚 No documents uploaded. Upload documents on the Home page or ask general questions that will be answered using web search.")
    
    # Chat configuration
    with st.expander("🔧 Chat Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            search_web = st.checkbox(
                "Enable web search fallback", 
                value=True,
                help="Search the web when questions are outside document scope"
            )
            
            max_context_length = st.slider(
                "Max context length",
                min_value=1000,
                max_value=4000,
                value=2000,
                help="Maximum characters of document context to include"
            )
        
        with col2:
            web_sources = st.multiselect(
                "Web search sources",
                ["DuckDuckGo", "Wikipedia", "arXiv"],
                default=["DuckDuckGo", "Wikipedia"],
                help="Select web sources for fallback search"
            )
    
    # Chat interface
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**{i+1}.** {source.get('title', source.get('metadata', {}).get('name', 'Unknown'))}")
                        if source.get('score'):
                            st.markdown(f"Relevance: {source['score']:.2f}")
                        if source.get('url'):
                            st.markdown(f"🔗 [Link]({source['url']})")
                        st.markdown("---")
    
    # Chat input
    user_input = st.chat_input("Ask a question about your documents or any topic...")
    
    if user_input:
        # Add user message to chat
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, sources = generate_chat_response(
                    user_input,
                    st.session_state.documents,
                    llm_engine,
                    search_engine,
                    search_web,
                    web_sources,
                    max_context_length
                )
            
            st.markdown(response)
            
            # Show sources
            if sources:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(sources):
                        st.markdown(f"**{i+1}.** {source.get('title', source.get('metadata', {}).get('name', 'Unknown'))}")
                        if source.get('score'):
                            st.markdown(f"Relevance: {source['score']:.2f}")
                        if source.get('url'):
                            st.markdown(f"🔗 [Link]({source['url']})")
                        st.markdown("---")
        
        # Add assistant response to chat
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "sources": sources
        })
        
        st.rerun()
    
    # Chat controls
    render_chat_controls()

def generate_chat_response(
    query: str,
    documents: List[Dict],
    llm_engine: LLMEngine,
    search_engine: SearchEngine,
    search_web: bool = True,
    web_sources: List[str] = None,
    max_context_length: int = 2000
) -> tuple[str, List[Dict]]:
    """Generate chat response with document search and web fallback"""
    
    sources = []
    context = ""
    
    try:
        # First, search documents if available
        if documents:
            doc_results = search_engine.search_documents(query, documents, top_k=3)
            
            if doc_results:
                # Build context from document results
                context_parts = []
                for result in doc_results:
                    content = result.get('content', '')
                    if len('\n'.join(context_parts) + content) < max_context_length:
                        context_parts.append(content)
                        sources.append(result)
                
                context = '\n\n'.join(context_parts)
        
        # If no good document results and web search enabled, search web
        if (not context or len(context) < 100) and search_web:
            try:
                web_results = search_engine.web_search(
                    query, 
                    sources=[source.lower() for source in web_sources] if web_sources else ['duckduckgo', 'wikipedia']
                )
                
                if web_results:
                    # Add web results to context
                    web_context_parts = []
                    for result in web_results[:2]:  # Limit to top 2 web results
                        content = result.get('content', '')
                        if content and len('\n'.join(web_context_parts) + content) < max_context_length:
                            web_context_parts.append(content)
                            sources.append(result)
                    
                    if web_context_parts:
                        web_context = '\n\n'.join(web_context_parts)
                        context = f"{context}\n\n{web_context}" if context else web_context
            
            except Exception as e:
                logger.error(f"Web search failed: {e}")
        
        # Generate response using LLM
        if context:
            prompt = f"""Based on the following information, please answer the user's question accurately and helpfully.

Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the available information. If the context doesn't fully answer the question, acknowledge what you can answer and what might require additional information."""
        else:
            prompt = f"""Please answer the following question to the best of your knowledge:

{query}

Note: No specific document context was found, so please provide a general informative response."""
        
        response = llm_engine.generate_response(prompt, context, task_type="chat")
        
        # Handle model failures
        if response.startswith("❌"):
            response = "I apologize, but I'm currently unable to process your question due to model availability issues. Please try again or contact support if the problem persists."
        
        return response, sources
    
    except Exception as e:
        logger.error(f"Chat response generation failed: {e}")
        return f"I apologize, but I encountered an error while processing your question: {str(e)}", []

def is_question_in_document_scope(query: str, documents: List[Dict]) -> bool:
    """Simple heuristic to determine if question might be answered by documents"""
    if not documents:
        return False
    
    # Get document keywords/topics
    doc_text = " ".join([doc.get('content', '')[:500] for doc in documents])
    query_words = query.lower().split()
    
    # Check for overlap
    overlap = sum(1 for word in query_words if word in doc_text.lower())
    return overlap > len(query_words) * 0.3  # 30% overlap threshold

def format_context_for_llm(search_results: List[Dict], max_length: int = 2000) -> str:
    """Format search results into context for LLM"""
    context_parts = []
    current_length = 0
    
    for result in search_results:
        content = result.get('content', '')
        source_name = result.get('metadata', {}).get('name', 'Unknown source')
        
        # Add source header
        header = f"\n--- From {source_name} ---\n"
        formatted_content = header + content
        
        if current_length + len(formatted_content) <= max_length:
            context_parts.append(formatted_content)
            current_length += len(formatted_content)
        else:
            # Add partial content if space allows
            remaining_space = max_length - current_length - len(header)
            if remaining_space > 100:
                partial_content = content[:remaining_space] + "..."
                context_parts.append(header + partial_content)
            break
    
    return '\n'.join(context_parts)
