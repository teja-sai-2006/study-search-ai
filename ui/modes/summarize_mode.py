import streamlit as st
from typing import List, Dict, Any
import logging
from utils.llm_engine import LLMEngine
from ui.components.file_uploader import render_document_selector
from ui.components.export_manager import render_quick_export_buttons

logger = logging.getLogger(__name__)

def render_summarize_mode():
    """Render summarize mode with customizable options"""
    st.markdown("## 📝 Summarize Mode")
    st.markdown('<div class="mode-description">Create intelligent summaries of your documents with customizable difficulty levels and output styles. Select multiple documents and choose your preferred summary format.</div>', unsafe_allow_html=True)
    
    # Check if documents are available
    if not st.session_state.documents:
        st.warning("📚 No documents available. Please upload documents on the Home page first.")
        return
    
    # Document selection
    selected_indices = render_document_selector(st.session_state.documents, "summarize")
    
    if not selected_indices:
        st.info("👆 Please select documents to summarize.")
        return
    
    # Summarization options
    st.markdown("### ⚙️ Summary Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        difficulty = st.selectbox(
            "📊 Difficulty Level",
            ["Beginner", "Intermediate", "Advanced"],
            help="Choose the complexity level for your summary"
        )
        
        include_images = st.checkbox(
            "🖼️ Include image descriptions",
            value=True,
            help="Include descriptions of charts, diagrams, and images found in documents"
        )
    
    with col2:
        styles = st.multiselect(
            "📋 Summary Styles",
            ["Paragraph", "Bullet Points", "Table", "Mind Map", "Key Points"],
            default=["Paragraph"],
            help="Select multiple summary formats"
        )
        
        max_length = st.slider(
            "📏 Summary Length",
            min_value=200,
            max_value=2000,
            value=800,
            step=100,
            help="Approximate number of words for the summary"
        )
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            focus_topics = st.text_area(
                "🎯 Focus Topics (optional)",
                placeholder="Enter specific topics to emphasize, separated by commas",
                help="Leave empty for general summary"
            )
            
            custom_instructions = st.text_area(
                "📝 Custom Instructions (optional)",
                placeholder="Any specific requirements for the summary",
                help="Additional instructions for the AI"
            )
        
        with col2:
            extract_key_terms = st.checkbox("🔑 Extract key terms", value=True)
            include_citations = st.checkbox("📖 Include source citations", value=True)
            merge_similar_content = st.checkbox("🔄 Merge similar content", value=True)
    
    # Generate summary
    if st.button("📝 Generate Summary", type="primary"):
        selected_docs = [st.session_state.documents[i] for i in selected_indices]
        
        with st.spinner("Generating summary..."):
            summaries = generate_summaries(
                selected_docs,
                styles,
                difficulty,
                max_length,
                include_images,
                focus_topics,
                custom_instructions,
                extract_key_terms,
                include_citations,
                merge_similar_content
            )
        
        if summaries:
            # Display summaries
            st.markdown("## 📋 Generated Summaries")
            
            for style, summary_content in summaries.items():
                with st.expander(f"📄 {style} Summary", expanded=True):
                    if style == "Table":
                        # Display table format
                        try:
                            import pandas as pd
                            if isinstance(summary_content, list):
                                df = pd.DataFrame(summary_content)
                                st.dataframe(df)
                            else:
                                st.markdown(summary_content)
                        except:
                            st.markdown(summary_content)
                    else:
                        st.markdown(summary_content)
            
            # Save to session state
            if 'summaries' not in st.session_state:
                st.session_state.summaries = []
            
            summary_entry = {
                'timestamp': pd.Timestamp.now(),
                'documents': [doc['name'] for doc in selected_docs],
                'summaries': summaries,
                'options': {
                    'difficulty': difficulty,
                    'styles': styles,
                    'max_length': max_length
                }
            }
            st.session_state.summaries.append(summary_entry)
            
            # Export options
            st.markdown("### 📤 Export Summary")
            render_quick_export_buttons(summaries, "summary")
            
        else:
            st.error("❌ Failed to generate summary. Please try again.")

def generate_summaries(
    documents: List[Dict],
    styles: List[str],
    difficulty: str,
    max_length: int,
    include_images: bool,
    focus_topics: str = "",
    custom_instructions: str = "",
    extract_key_terms: bool = False,
    include_citations: bool = False,
    merge_similar_content: bool = False
) -> Dict[str, str]:
    """Generate summaries in multiple styles"""
    
    llm_engine = LLMEngine()
    summaries = {}
    
    try:
        # Prepare document content
        content_blocks = []
        for doc in documents:
            content = doc.get('content', '')
            if content:
                doc_name = doc.get('name', 'Unknown Document')
                content_blocks.append(f"=== {doc_name} ===\n{content}")
        
        if not content_blocks:
            return {}
        
        combined_content = '\n\n'.join(content_blocks)
        
        # Process each style
        for style in styles:
            try:
                summary = generate_single_summary(
                    combined_content,
                    style,
                    difficulty,
                    max_length,
                    focus_topics,
                    custom_instructions,
                    extract_key_terms,
                    include_citations,
                    merge_similar_content,
                    llm_engine
                )
                summaries[style] = summary
            
            except Exception as e:
                logger.error(f"Failed to generate {style} summary: {e}")
                summaries[style] = f"❌ Failed to generate {style} summary: {str(e)}"
        
        return summaries
    
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        return {}

def generate_single_summary(
    content: str,
    style: str,
    difficulty: str,
    max_length: int,
    focus_topics: str,
    custom_instructions: str,
    extract_key_terms: bool,
    include_citations: bool,
    merge_similar_content: bool,
    llm_engine: LLMEngine
) -> str:
    """Generate a single summary in specified style"""
    
    # Build prompt based on style and options
    prompt = f"""Please create a {style.lower()} summary of the following documents.

Requirements:
- Difficulty level: {difficulty}
- Target length: approximately {max_length} words
- Style: {style}
"""
    
    if focus_topics:
        prompt += f"- Focus particularly on these topics: {focus_topics}\n"
    
    if custom_instructions:
        prompt += f"- Additional instructions: {custom_instructions}\n"
    
    # Style-specific instructions
    if style == "Paragraph":
        prompt += """
Format as coherent paragraphs with clear topic transitions.
Start with an overview, then cover main topics in logical order.
"""
    elif style == "Bullet Points":
        prompt += """
Format as a hierarchical bullet point structure:
- Main topics as primary bullets
  - Supporting details as sub-bullets
    - Specific examples as tertiary bullets
"""
    elif style == "Table":
        prompt += """
Format as a structured table with columns like:
| Topic | Key Points | Details | Source |
Use markdown table format.
"""
    elif style == "Mind Map":
        prompt += """
Format as a text-based mind map structure:
CENTRAL TOPIC
├── Branch 1
│   ├── Sub-topic 1.1
│   └── Sub-topic 1.2
└── Branch 2
    ├── Sub-topic 2.1
    └── Sub-topic 2.2
"""
    elif style == "Key Points":
        prompt += """
Format as numbered key points with brief explanations:
1. KEY POINT: Brief explanation
2. KEY POINT: Brief explanation
"""
    
    # Additional requirements
    if extract_key_terms:
        prompt += "- Include a section with key terms and definitions\n"
    
    if include_citations:
        prompt += "- Include references to source documents when making specific claims\n"
    
    if merge_similar_content:
        prompt += "- Merge and consolidate similar information from different sources\n"
    
    # Difficulty-specific instructions
    if difficulty == "Beginner":
        prompt += """
- Use simple, clear language
- Explain technical terms
- Focus on main concepts rather than details
"""
    elif difficulty == "Intermediate":
        prompt += """
- Use moderate complexity language
- Include some technical details
- Balance concepts with practical applications
"""
    elif difficulty == "Advanced":
        prompt += """
- Use technical language appropriate for experts
- Include detailed analysis and nuanced points
- Focus on implications and advanced concepts
"""
    
    prompt += f"\n\nDocument content:\n{content[:8000]}..."  # Limit content length
    
    # Generate summary
    response = llm_engine.generate_response(prompt, task_type="summarize")
    
    return response

def display_summary_history():
    """Display previously generated summaries"""
    if 'summaries' not in st.session_state or not st.session_state.summaries:
        st.info("No previous summaries found.")
        return
    
    st.markdown("### 📚 Summary History")
    
    for i, summary_entry in enumerate(reversed(st.session_state.summaries)):
        with st.expander(f"Summary {len(st.session_state.summaries) - i} - {summary_entry['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
            st.markdown(f"**Documents:** {', '.join(summary_entry['documents'])}")
            st.markdown(f"**Difficulty:** {summary_entry['options']['difficulty']}")
            st.markdown(f"**Styles:** {', '.join(summary_entry['options']['styles'])}")
            
            for style, content in summary_entry['summaries'].items():
                with st.expander(f"📄 {style}"):
                    st.markdown(content)

# Add to the main render function
def render_summarize_mode():
    """Enhanced render function with history"""
    # ... existing code ...
    
    # Add tab for history
    tab1, tab2 = st.tabs(["📝 Create Summary", "📚 History"])
    
    with tab1:
        # ... existing summarization code ...
        pass
    
    with tab2:
        display_summary_history()
