import streamlit as st
from typing import List, Dict, Any, Optional
import logging
from utils.image_processor import ImageProcessor
from utils.document_processor import DocumentProcessor
from ui.components.export_manager import render_quick_export_buttons

logger = logging.getLogger(__name__)

def render_image_mode():
    """Render image mode for image processing and analysis"""
    st.markdown("## 🖼️ Image Mode")
    st.markdown('<div class="mode-description">Upload images or select from PDF documents to extract text, generate captions, and analyze educational content. Perfect for processing diagrams, charts, and scanned documents.</div>', unsafe_allow_html=True)
    
    # Initialize image processor
    image_processor = ImageProcessor()
    
    # Image source selection
    st.markdown("### 📷 Image Source")
    
    source_option = st.radio(
        "Choose image source:",
        ["📤 Upload new image", "📄 Select from PDF documents"],
        help="Choose whether to upload a new image or use images from uploaded PDFs"
    )
    
    selected_images = []
    
    if source_option == "📤 Upload new image":
        uploaded_images = st.file_uploader(
            "Upload images",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            accept_multiple_files=True,
            help="Supported formats: PNG, JPG, JPEG, BMP, TIFF"
        )
        
        if uploaded_images:
            selected_images = uploaded_images
            
            # Show preview
            st.markdown("#### 👁️ Image Preview")
            cols = st.columns(min(3, len(uploaded_images)))
            for i, img in enumerate(uploaded_images):
                with cols[i % len(cols)]:
                    st.image(img, caption=img.name, use_column_width=True)
    
    else:  # Select from PDF documents
        if not st.session_state.documents:
            st.warning("📚 No documents available. Please upload PDF documents on the Home page first.")
            return
        
        # Filter PDF documents
        pdf_docs = [doc for doc in st.session_state.documents if doc.get('format') == 'pdf']
        
        if not pdf_docs:
            st.warning("📄 No PDF documents found. This feature requires PDF documents with images.")
            return
        
        selected_pdf = st.selectbox(
            "Select PDF document:",
            range(len(pdf_docs)),
            format_func=lambda i: pdf_docs[i].get('name', f'PDF {i+1}')
        )
        
        if st.button("🔍 Extract Images from PDF"):
            with st.spinner("Extracting images from PDF..."):
                # This would need the actual PDF file path
                # For now, we'll simulate the process
                st.info("🔧 PDF image extraction feature requires file system access. Please upload individual images for now.")
    
    # Processing options
    st.markdown("### ⚙️ Processing Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        extract_text = st.checkbox(
            "📝 Extract text (OCR)",
            value=True,
            help="Extract text from images using OCR"
        )
        
        generate_caption = st.checkbox(
            "💬 Generate caption",
            value=True,
            help="Generate descriptive captions using AI"
        )
        
        analyze_educational = st.checkbox(
            "🎓 Educational analysis",
            value=True,
            help="Analyze for educational content like charts, diagrams, formulas"
        )
    
    with col2:
        ocr_language = st.selectbox(
            "🌍 OCR Language",
            ["English", "Spanish", "French", "German", "Multi-language"],
            help="Select language for OCR processing"
        )
        
        detail_level = st.selectbox(
            "📊 Analysis Detail",
            ["Basic", "Detailed", "Comprehensive"],
            help="Choose level of detail for analysis"
        )
        
        include_confidence = st.checkbox(
            "📈 Include confidence scores",
            value=False,
            help="Show confidence scores for OCR and analysis"
        )
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            enhance_image = st.checkbox(
                "✨ Enhance image quality",
                value=True,
                help="Preprocess images to improve OCR accuracy"
            )
            
            detect_tables = st.checkbox(
                "📊 Detect tables/charts",
                value=True,
                help="Specifically analyze tabular data and charts"
            )
        
        with col2:
            extract_formulas = st.checkbox(
                "🧮 Extract formulas",
                value=True,
                help="Identify and extract mathematical formulas"
            )
            
            custom_instructions = st.text_area(
                "📝 Custom analysis instructions",
                placeholder="Any specific instructions for image analysis...",
                help="Provide specific instructions for analysis"
            )
    
    # Process images
    if selected_images and st.button("🔍 Process Images", type="primary"):
        process_images(
            selected_images,
            image_processor,
            extract_text,
            generate_caption,
            analyze_educational,
            ocr_language,
            detail_level,
            include_confidence,
            enhance_image,
            detect_tables,
            extract_formulas,
            custom_instructions
        )

def process_images(
    images: List,
    image_processor: ImageProcessor,
    extract_text: bool,
    generate_caption: bool,
    analyze_educational: bool,
    ocr_language: str,
    detail_level: str,
    include_confidence: bool,
    enhance_image: bool,
    detect_tables: bool,
    extract_formulas: bool,
    custom_instructions: str
):
    """Process uploaded images with specified options"""
    
    st.markdown("## 🔍 Image Analysis Results")
    
    all_results = []
    
    for i, image in enumerate(images):
        st.markdown(f"### 📷 Image {i+1}: {image.name}")
        
        with st.spinner(f"Processing {image.name}..."):
            try:
                # Basic processing
                result = image_processor.process_image(
                    image,
                    extract_text=extract_text,
                    generate_caption=generate_caption
                )
                
                if result.get('error'):
                    st.error(f"❌ Error processing {image.name}: {result['error']}")
                    continue
                
                # Display image
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(image, caption=image.name, use_column_width=True)
                
                with col2:
                    # Basic results
                    if result.get('caption'):
                        st.markdown("#### 💬 AI Caption")
                        st.info(result['caption'])
                    
                    if result.get('text') and not result['text'].startswith("❌"):
                        st.markdown("#### 📝 Extracted Text")
                        st.text_area(
                            "OCR Text:",
                            result['text'],
                            height=100,
                            disabled=True,
                            key=f"ocr_{i}"
                        )
                    elif extract_text:
                        st.warning("No text found in image")
                
                # Educational analysis
                if analyze_educational and result.get('caption'):
                    with st.spinner("Performing educational analysis..."):
                        educational_analysis = perform_educational_analysis(
                            image,
                            result,
                            detail_level,
                            detect_tables,
                            extract_formulas,
                            custom_instructions,
                            image_processor
                        )
                        
                        if educational_analysis:
                            st.markdown("#### 🎓 Educational Analysis")
                            st.markdown(educational_analysis)
                
                # Confidence scores
                if include_confidence and extract_text:
                    confidence_data = get_confidence_scores(image, image_processor)
                    if confidence_data:
                        st.markdown("#### 📈 Confidence Scores")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("OCR Confidence", f"{confidence_data.get('confidence', 0):.1f}%")
                        with col2:
                            st.metric("Words Detected", confidence_data.get('word_count', 0))
                        with col3:
                            st.metric("Text Quality", get_quality_rating(confidence_data.get('confidence', 0)))
                
                # Store results
                complete_result = {
                    'filename': image.name,
                    'caption': result.get('caption', ''),
                    'text': result.get('text', ''),
                    'educational_analysis': educational_analysis if analyze_educational else '',
                    'confidence': confidence_data if include_confidence else None
                }
                
                all_results.append(complete_result)
                
                st.markdown("---")
            
            except Exception as e:
                st.error(f"❌ Failed to process {image.name}: {str(e)}")
                logger.error(f"Image processing failed: {e}")
    
    # Save results and show export options
    if all_results:
        # Save to session state
        if 'image_analysis_results' not in st.session_state:
            st.session_state.image_analysis_results = []
        
        st.session_state.image_analysis_results.extend(all_results)
        
        # Export options
        st.markdown("### 📤 Export Results")
        export_data = {
            'image_analysis': all_results,
            'processing_options': {
                'extract_text': extract_text,
                'generate_caption': generate_caption,
                'analyze_educational': analyze_educational,
                'detail_level': detail_level
            }
        }
        render_quick_export_buttons(export_data, "image_analysis")
        
        st.success(f"✅ Successfully processed {len(all_results)} images!")

def perform_educational_analysis(
    image,
    basic_result: Dict[str, Any],
    detail_level: str,
    detect_tables: bool,
    extract_formulas: bool,
    custom_instructions: str,
    image_processor: ImageProcessor
) -> str:
    """Perform detailed educational analysis of image"""
    
    try:
        from utils.llm_engine import LLMEngine
        llm_engine = LLMEngine()
        
        caption = basic_result.get('caption', '')
        text = basic_result.get('text', '')
        
        # Build analysis prompt
        prompt = f"""Analyze this educational image based on the following information:

Image Caption: {caption}
Extracted Text: {text}

Please provide educational analysis covering:
"""
        
        if detail_level == "Basic":
            prompt += """
1. Type of educational content (diagram, chart, text, etc.)
2. Main topic or subject area
3. Key information presented
4. Educational value and context
"""
        elif detail_level == "Detailed":
            prompt += """
1. Detailed content type analysis
2. Subject area and educational level
3. Key concepts and information
4. Learning objectives that could be met
5. How this could be used in studying
6. Connections to broader topics
"""
        elif detail_level == "Comprehensive":
            prompt += """
1. Comprehensive content analysis
2. Educational taxonomy classification
3. Detailed concept mapping
4. Learning objectives and outcomes
5. Pedagogical applications
6. Assessment possibilities
7. Integration with curriculum
8. Prerequisite knowledge required
"""
        
        if detect_tables:
            prompt += "\n- Special attention to any tables, charts, or data visualizations"
        
        if extract_formulas:
            prompt += "\n- Identify and explain any mathematical formulas or equations"
        
        if custom_instructions:
            prompt += f"\n- Additional focus: {custom_instructions}"
        
        analysis = llm_engine.generate_response(prompt, task_type="analyze")
        return analysis
    
    except Exception as e:
        logger.error(f"Educational analysis failed: {e}")
        return f"Educational analysis unavailable: {str(e)}"

def get_confidence_scores(image, image_processor: ImageProcessor) -> Optional[Dict[str, Any]]:
    """Get OCR confidence scores for image"""
    
    try:
        confidence_data = image_processor._ocr_processor.extract_text_with_confidence(image)
        return confidence_data
    except Exception as e:
        logger.error(f"Confidence score extraction failed: {e}")
        return None

def get_quality_rating(confidence: float) -> str:
    """Convert confidence score to quality rating"""
    if confidence >= 90:
        return "Excellent"
    elif confidence >= 75:
        return "Good"
    elif confidence >= 60:
        return "Fair"
    elif confidence >= 40:
        return "Poor"
    else:
        return "Very Poor"

def display_image_history():
    """Display previously analyzed images"""
    if 'image_analysis_results' not in st.session_state or not st.session_state.image_analysis_results:
        st.info("No previous image analyses found.")
        return
    
    st.markdown("### 📚 Image Analysis History")
    
    for i, result in enumerate(st.session_state.image_analysis_results):
        with st.expander(f"📷 {result['filename']}"):
            if result.get('caption'):
                st.markdown(f"**Caption:** {result['caption']}")
            
            if result.get('text'):
                st.markdown("**Extracted Text:**")
                st.text(result['text'][:200] + "..." if len(result['text']) > 200 else result['text'])
            
            if result.get('educational_analysis'):
                st.markdown("**Educational Analysis:**")
                st.text(result['educational_analysis'][:300] + "..." if len(result['educational_analysis']) > 300 else result['educational_analysis'])
            
            if result.get('confidence'):
                st.markdown(f"**OCR Confidence:** {result['confidence'].get('confidence', 0):.1f}%")
