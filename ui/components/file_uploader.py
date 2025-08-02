import streamlit as st
from typing import List, Dict, Any, Optional

def render_file_uploader(
    label: str = "Upload files",
    file_types: List[str] = None,
    multiple: bool = True,
    help_text: str = None
) -> Optional[List]:
    """Enhanced file uploader with preview"""
    
    if file_types is None:
        file_types = ['pdf', 'docx', 'txt', 'md', 'rtf', 'xlsx', 'xls', 'csv', 'zip', 'png', 'jpg', 'jpeg']
    
    uploaded_files = st.file_uploader(
        label,
        type=file_types,
        accept_multiple_files=multiple,
        help=help_text or f"Supported formats: {', '.join(file_types).upper()}"
    )
    
    if uploaded_files:
        # Show file preview
        if multiple and isinstance(uploaded_files, list):
            with st.expander(f"📁 {len(uploaded_files)} files selected"):
                for file in uploaded_files:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"📄 {file.name}")
                    with col2:
                        st.write(f"{file.size} bytes")
                    with col3:
                        st.write(file.type)
        else:
            if not isinstance(uploaded_files, list):
                uploaded_files = [uploaded_files]
            
            with st.expander("📄 File details"):
                file = uploaded_files[0]
                st.write(f"**Name:** {file.name}")
                st.write(f"**Size:** {file.size} bytes")
                st.write(f"**Type:** {file.type}")
    
    return uploaded_files

def render_document_selector(documents: List[Dict[str, Any]], key: str = "doc_selector") -> List[int]:
    """Render document selector for multiple documents"""
    if not documents:
        st.warning("No documents available. Please upload documents first.")
        return []
    
    st.markdown("### 📚 Select Documents")
    
    # Select all option
    select_all = st.checkbox("Select all documents", key=f"{key}_select_all")
    
    if select_all:
        return list(range(len(documents)))
    
    # Individual selection
    selected_indices = []
    
    for i, doc in enumerate(documents):
        if st.checkbox(
            f"📄 {doc.get('name', f'Document {i+1}')} ({doc.get('format', 'unknown').upper()})",
            key=f"{key}_doc_{i}"
        ):
            selected_indices.append(i)
    
    return selected_indices

def render_folder_structure(files: List[str], title: str = "Folder Structure"):
    """Render folder structure visualization"""
    st.markdown(f"### 📁 {title}")
    
    # Group files by extension
    by_extension = {}
    for file in files:
        ext = file.split('.')[-1].lower() if '.' in file else 'unknown'
        if ext not in by_extension:
            by_extension[ext] = []
        by_extension[ext].append(file)
    
    # Display grouped files
    for ext, file_list in by_extension.items():
        with st.expander(f"📋 {ext.upper()} files ({len(file_list)})"):
            for file in file_list:
                st.write(f"📄 {file}")

def render_upload_progress(current: int, total: int, message: str = "Processing"):
    """Render upload progress bar"""
    progress = current / total if total > 0 else 0
    st.progress(progress, text=f"{message}: {current}/{total}")

def render_file_preview(file_content: str, file_type: str, max_length: int = 500):
    """Render file content preview"""
    if not file_content:
        st.warning("No content to preview")
        return
    
    with st.expander("👁️ Content Preview"):
        if len(file_content) > max_length:
            preview = file_content[:max_length] + "..."
            st.text_area("Preview", preview, height=200, disabled=True)
            st.info(f"Showing first {max_length} characters of {len(file_content)} total characters")
        else:
            st.text_area("Full Content", file_content, height=200, disabled=True)
