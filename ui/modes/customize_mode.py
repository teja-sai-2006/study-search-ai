import streamlit as st
from typing import Dict, Any
import logging
from utils.llm_engine import LLMEngine
from ui.components.file_uploader import render_document_selector
from ui.components.export_manager import render_quick_export_buttons

logger = logging.getLogger(__name__)

def render_customize_mode():
    """Render customize mode for custom content processing"""
    st.markdown("## 🛠️ Customize Mode")
    st.markdown('<div class="mode-description">Transform your documents or text with custom tone, depth, and formatting. Perfect for adapting content to different audiences or purposes.</div>', unsafe_allow_html=True)
    
    # Input selection
    st.markdown("### 📝 Content Input")
    
    input_type = st.radio(
        "Choose input source:",
        ["📄 Select from uploaded documents", "✍️ Paste custom text"],
        help="Choose whether to work with existing documents or enter new text"
    )
    
    selected_content = ""
    source_name = ""
    
    if input_type == "📄 Select from uploaded documents":
        if not st.session_state.documents:
            st.warning("📚 No documents available. Please upload documents on the Home page first.")
            return
        
        selected_indices = render_document_selector(st.session_state.documents, "customize")
        
        if selected_indices:
            # Combine selected documents
            selected_docs = [st.session_state.documents[i] for i in selected_indices]
            content_parts = []
            doc_names = []
            
            for doc in selected_docs:
                content = doc.get('content', '')
                if isinstance(content, str) and content:
                    content_parts.append(content)
                    doc_names.append(doc.get('name', 'Unknown'))
                elif isinstance(content, dict):
                    # Handle case where content is a dict - extract text
                    content_text = str(content.get('text', content.get('content', '')))
                    if content_text:
                        content_parts.append(content_text)
                        doc_names.append(doc.get('name', 'Unknown'))
                elif content:
                    # Convert any other type to string
                    content_parts.append(str(content))
                    doc_names.append(doc.get('name', 'Unknown'))
            
            selected_content = '\n\n'.join(content_parts)
            source_name = ', '.join(doc_names)
        
        if not selected_content:
            st.info("👆 Please select documents to customize.")
            return
    
    else:  # Custom text input
        selected_content = st.text_area(
            "📝 Enter your text:",
            height=200,
            placeholder="Paste or type your content here...",
            help="Enter the text you want to transform"
        )
        source_name = "Custom Text"
        
        if not selected_content.strip():
            st.info("👆 Please enter text to customize.")
            return
    
    # Show content preview
    if selected_content:
        with st.expander("👁️ Content Preview"):
            preview_length = min(500, len(selected_content))
            st.text_area(
                f"First {preview_length} characters:",
                selected_content[:preview_length] + ("..." if len(selected_content) > preview_length else ""),
                height=100,
                disabled=True
            )
            st.caption(f"Total length: {len(selected_content)} characters")
    
    # Customization options
    st.markdown("### ⚙️ Customization Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        tone = st.selectbox(
            "🎭 Tone",
            ["Simple", "Formal", "Clear", "Academic", "Conversational", "Professional", "Friendly"],
            help="Choose the tone for the customized content"
        )
        
        depth = st.selectbox(
            "📊 Depth",
            ["Brief", "In-depth", "Comprehensive", "Summary", "Detailed Analysis"],
            help="Choose how detailed the output should be"
        )
        
        target_audience = st.selectbox(
            "👥 Target Audience",
            ["General", "Students", "Professionals", "Experts", "Beginners", "Children"],
            help="Who is the intended audience?"
        )
    
    with col2:
        output_format = st.selectbox(
            "📋 Output Format",
            ["Markdown", "Plain Text", "HTML", "Structured List", "Q&A Format", "Step-by-Step Guide"],
            help="Choose the output format"
        )
        
        length_preference = st.selectbox(
            "📏 Length",
            ["Keep Original", "Shorter", "Longer", "Much Shorter", "Much Longer"],
            help="Adjust the length relative to original"
        )
        
        include_examples = st.checkbox(
            "💡 Include examples",
            value=False,
            help="Add relevant examples and illustrations"
        )
    
    # Advanced customization
    with st.expander("🔧 Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            custom_instructions = st.text_area(
                "📝 Custom Instructions",
                placeholder="Any specific requirements or instructions...",
                help="Provide specific instructions for customization"
            )
            
            preserve_structure = st.checkbox(
                "🏗️ Preserve original structure",
                value=True,
                help="Keep the general organization of the original content"
            )
        
        with col2:
            add_headings = st.checkbox(
                "📑 Add section headings",
                value=True,
                help="Organize content with clear headings"
            )
            
            include_summary = st.checkbox(
                "📋 Include summary",
                value=False,
                help="Add a summary at the beginning or end"
            )
            
            highlight_key_points = st.checkbox(
                "⭐ Highlight key points",
                value=True,
                help="Emphasize important information"
            )
    
    # Generate customized content
    if st.button("🛠️ Customize Content", type="primary"):
        with st.spinner("Customizing content..."):
            customized_content = generate_customized_content(
                selected_content,
                tone,
                depth,
                target_audience,
                output_format,
                length_preference,
                include_examples,
                custom_instructions,
                preserve_structure,
                add_headings,
                include_summary,
                highlight_key_points
            )
        
        if customized_content and not customized_content.startswith("❌"):
            # Display result
            st.markdown("## ✨ Customized Content")
            
            # Show options tabs
            tab1, tab2 = st.tabs(["📖 Formatted View", "📝 Raw Text"])
            
            with tab1:
                if output_format == "Markdown":
                    st.markdown(customized_content)
                elif output_format == "HTML":
                    st.components.v1.html(customized_content, height=400, scrolling=True)
                else:
                    st.text(customized_content)
            
            with tab2:
                st.text_area(
                    "Raw customized content:",
                    customized_content,
                    height=400,
                    disabled=True
                )
            
            # Save to session state
            if 'customized_content' not in st.session_state:
                st.session_state.customized_content = []
            
            entry = {
                'timestamp': pd.Timestamp.now(),
                'source': source_name,
                'content': customized_content,
                'options': {
                    'tone': tone,
                    'depth': depth,
                    'audience': target_audience,
                    'format': output_format
                }
            }
            st.session_state.customized_content.append(entry)
            
            # Export options
            st.markdown("### 📤 Export Customized Content")
            export_content = {
                'customized_content': customized_content,
                'original_source': source_name,
                'customization_options': entry['options']
            }
            render_quick_export_buttons(export_content, "customize")
            
        else:
            st.error("❌ Failed to customize content. Please try again or adjust your settings.")

def generate_customized_content(
    content: str,
    tone: str,
    depth: str,
    target_audience: str,
    output_format: str,
    length_preference: str,
    include_examples: bool,
    custom_instructions: str,
    preserve_structure: bool,
    add_headings: bool,
    include_summary: bool,
    highlight_key_points: bool
) -> str:
    """Generate customized content based on specifications"""
    
    llm_engine = LLMEngine()
    
    try:
        # Build comprehensive prompt
        prompt = f"""Please customize the following content according to these specifications:

TONE: {tone}
DEPTH: {depth}
TARGET AUDIENCE: {target_audience}
OUTPUT FORMAT: {output_format}
LENGTH: {length_preference}

"""
        
        # Add formatting requirements
        if output_format == "Markdown":
            prompt += """
FORMAT REQUIREMENTS:
- Use proper Markdown syntax
- Include headers (##, ###) for organization
- Use **bold** and *italic* for emphasis
- Use lists where appropriate
- Ensure clean, readable formatting
"""
        elif output_format == "HTML":
            prompt += """
FORMAT REQUIREMENTS:
- Use proper HTML tags
- Include <h2>, <h3> headers for organization
- Use <strong> and <em> for emphasis
- Use <ul> and <ol> for lists
- Ensure valid HTML structure
"""
        elif output_format == "Structured List":
            prompt += """
FORMAT REQUIREMENTS:
- Organize as numbered or bulleted lists
- Use hierarchical structure
- Each main point should be clearly separated
"""
        elif output_format == "Q&A Format":
            prompt += """
FORMAT REQUIREMENTS:
- Present information as questions and answers
- Start each section with a clear question
- Provide comprehensive answers
"""
        elif output_format == "Step-by-Step Guide":
            prompt += """
FORMAT REQUIREMENTS:
- Break down into sequential steps
- Number each step clearly
- Include sub-steps where necessary
- Make each step actionable
"""
        
        # Add specific requirements
        requirements = []
        
        if preserve_structure:
            requirements.append("- Maintain the general structure and flow of the original content")
        
        if add_headings:
            requirements.append("- Add clear section headings to organize the content")
        
        if include_summary:
            requirements.append("- Include a brief summary at the beginning")
        
        if highlight_key_points:
            requirements.append("- Emphasize and highlight the most important points")
        
        if include_examples:
            requirements.append("- Add relevant examples and illustrations to clarify concepts")
        
        if custom_instructions:
            requirements.append(f"- Follow these specific instructions: {custom_instructions}")
        
        # Length adjustments
        if length_preference == "Shorter":
            requirements.append("- Make the content more concise while preserving key information")
        elif length_preference == "Longer":
            requirements.append("- Expand the content with additional details and explanations")
        elif length_preference == "Much Shorter":
            requirements.append("- Significantly condense the content to essential points only")
        elif length_preference == "Much Longer":
            requirements.append("- Substantially expand with comprehensive details, examples, and analysis")
        
        if requirements:
            prompt += "\nADDITIONAL REQUIREMENTS:\n" + "\n".join(requirements) + "\n"
        
        # Audience-specific instructions
        audience_instructions = {
            "Students": "Use educational language, include learning objectives, and structure for study purposes",
            "Professionals": "Use professional terminology, focus on practical applications and business implications",
            "Experts": "Use technical language, include advanced concepts and detailed analysis",
            "Beginners": "Use simple language, explain technical terms, and provide step-by-step guidance",
            "Children": "Use simple words, include engaging elements, and break down complex ideas",
            "General": "Use accessible language suitable for a broad audience"
        }
        
        if target_audience in audience_instructions:
            prompt += f"\nAUDIENCE CONSIDERATIONS: {audience_instructions[target_audience]}\n"
        
        prompt += f"\nORIGINAL CONTENT:\n{content[:6000]}..."  # Limit content length
        
        # Generate customized content
        response = llm_engine.generate_response(prompt, task_type="generate")
        
        return response
    
    except Exception as e:
        logger.error(f"Content customization failed: {e}")
        return f"❌ Failed to customize content: {str(e)}"

def display_customization_history():
    """Display previously customized content"""
    if 'customized_content' not in st.session_state or not st.session_state.customized_content:
        st.info("No previous customizations found.")
        return
    
    st.markdown("### 📚 Customization History")
    
    for i, entry in enumerate(reversed(st.session_state.customized_content)):
        with st.expander(f"Customization {len(st.session_state.customized_content) - i} - {entry['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
            st.markdown(f"**Source:** {entry['source']}")
            st.markdown(f"**Tone:** {entry['options']['tone']}")
            st.markdown(f"**Depth:** {entry['options']['depth']}")
            st.markdown(f"**Audience:** {entry['options']['audience']}")
            st.markdown(f"**Format:** {entry['options']['format']}")
            
            with st.expander("📖 View Content"):
                if entry['options']['format'] == "Markdown":
                    st.markdown(entry['content'])
                else:
                    st.text(entry['content'])
