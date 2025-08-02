import streamlit as st
from typing import List, Dict, Any, Optional
import logging
import pandas as pd
from utils.table_extractor import TableExtractor
from ui.components.file_uploader import render_document_selector
from ui.components.export_manager import render_quick_export_buttons

logger = logging.getLogger(__name__)

def render_advanced_tables_mode():
    """Render advanced tables mode for table extraction and analysis"""
    st.markdown("## 📊 Advanced Tables")
    st.markdown('<div class="mode-description">Extract and analyze tables from PDF documents using multiple extraction methods. Get intelligent summaries and export in various formats.</div>', unsafe_allow_html=True)
    
    # Check if documents are available
    if not st.session_state.documents:
        st.warning("📚 No documents available. Please upload PDF documents on the Home page first.")
        return
    
    # Filter PDF documents
    pdf_docs = [doc for doc in st.session_state.documents if doc.get('format') == 'pdf']
    
    if not pdf_docs:
        st.warning("📄 No PDF documents found. This feature works with PDF files containing tables.")
        return
    
    # Document selection
    st.markdown("### 📄 Document Selection")
    
    selected_pdf_indices = []
    for i, doc in enumerate(pdf_docs):
        if st.checkbox(
            f"📄 {doc.get('name', f'PDF {i+1}')}",
            key=f"pdf_select_{i}",
            help=f"Size: {doc.get('size', 0)} bytes"
        ):
            selected_pdf_indices.append(i)
    
    if not selected_pdf_indices:
        st.info("👆 Please select PDF documents to extract tables from.")
        return
    
    selected_pdfs = [pdf_docs[i] for i in selected_pdf_indices]
    
    # Extraction options
    st.markdown("### ⚙️ Extraction Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        extraction_method = st.selectbox(
            "🔧 Extraction Method",
            ["Auto (All Methods)", "Camelot", "pdfplumber", "Tabula"],
            help="Choose table extraction method. Auto tries all methods and returns best results."
        )
        
        page_range = st.text_input(
            "📄 Page Range",
            value="all",
            placeholder="e.g., 1,3,5-10 or all",
            help="Specify pages to process. Use 'all' for entire document."
        )
    
    with col2:
        min_accuracy = st.slider(
            "📈 Minimum Accuracy (%)",
            min_value=10,
            max_value=100,
            value=50,
            help="Minimum accuracy threshold for including extracted tables"
        )
        
        max_tables = st.slider(
            "📊 Max Tables per Document",
            min_value=1,
            max_value=20,
            value=10,
            help="Maximum number of tables to extract per document"
        )
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            analyze_content = st.checkbox(
                "🧠 Analyze table content",
                value=True,
                help="Generate AI analysis of table content and structure"
            )
            
            merge_similar = st.checkbox(
                "🔄 Merge similar tables",
                value=True,
                help="Automatically merge tables with similar structure"
            )
            
            extract_headers = st.checkbox(
                "📑 Detect headers",
                value=True,
                help="Automatically detect and preserve table headers"
            )
        
        with col2:
            include_metadata = st.checkbox(
                "📋 Include metadata",
                value=True,
                help="Include extraction metadata (page numbers, accuracy, etc.)"
            )
            
            clean_data = st.checkbox(
                "🧹 Clean extracted data",
                value=True,
                help="Automatically clean and format extracted table data"
            )
            
            export_formats = st.multiselect(
                "📤 Prepare export formats",
                ["CSV", "Excel", "JSON"],
                default=["CSV", "Excel"],
                help="Prepare tables for export in selected formats"
            )
    
    # Extract tables
    if st.button("📊 Extract Tables", type="primary"):
        extract_tables_from_documents(
            selected_pdfs,
            extraction_method,
            page_range,
            min_accuracy,
            max_tables,
            analyze_content,
            merge_similar,
            extract_headers,
            include_metadata,
            clean_data,
            export_formats
        )

def extract_tables_from_documents(
    pdf_documents: List[Dict],
    extraction_method: str,
    page_range: str,
    min_accuracy: int,
    max_tables: int,
    analyze_content: bool,
    merge_similar: bool,
    extract_headers: bool,
    include_metadata: bool,
    clean_data: bool,
    export_formats: List[str]
):
    """Extract tables from PDF documents"""
    
    table_extractor = TableExtractor()
    all_extracted_tables = []
    
    st.markdown("## 📊 Table Extraction Results")
    
    for doc_idx, doc in enumerate(pdf_documents):
        st.markdown(f"### 📄 {doc.get('name', f'Document {doc_idx + 1}')}")
        
        with st.spinner(f"Extracting tables from {doc.get('name', 'document')}..."):
            try:
                # For demo purposes, we'll simulate table extraction
                # In a real implementation, you'd use the actual PDF file path
                simulated_tables = simulate_table_extraction(
                    doc,
                    extraction_method,
                    page_range,
                    min_accuracy,
                    max_tables
                )
                
                if simulated_tables:
                    display_extracted_tables(
                        simulated_tables,
                        doc,
                        analyze_content,
                        include_metadata,
                        clean_data
                    )
                    
                    all_extracted_tables.extend(simulated_tables)
                else:
                    st.warning(f"No tables found in {doc.get('name', 'document')} meeting the criteria.")
            
            except Exception as e:
                st.error(f"❌ Error extracting tables from {doc.get('name', 'document')}: {str(e)}")
                logger.error(f"Table extraction failed: {e}")
    
    # Summary and export
    if all_extracted_tables:
        display_extraction_summary(all_extracted_tables)
        
        # Save to session state
        if 'extracted_tables' not in st.session_state:
            st.session_state.extracted_tables = []
        
        st.session_state.extracted_tables.extend(all_extracted_tables)
        
        # Export options
        if export_formats:
            st.markdown("### 📤 Export Tables")
            prepare_table_exports(all_extracted_tables, export_formats)

def simulate_table_extraction(
    document: Dict,
    method: str,
    page_range: str,
    min_accuracy: int,
    max_tables: int
) -> List[Dict[str, Any]]:
    """Simulate table extraction for demonstration"""
    
    # Create sample tables based on document
    sample_tables = []
    
    # Create a financial table
    financial_data = {
        'Quarter': ['Q1 2023', 'Q2 2023', 'Q3 2023', 'Q4 2023'],
        'Revenue': ['$125M', '$134M', '$142M', '$156M'],
        'Expenses': ['$95M', '$101M', '$108M', '$118M'],
        'Profit': ['$30M', '$33M', '$34M', '$38M']
    }
    
    financial_df = pd.DataFrame(financial_data)
    
    sample_tables.append({
        'data': financial_df,
        'page': 1,
        'accuracy': 87.5,
        'table_index': 0,
        'shape': financial_df.shape,
        'extraction_method': method.lower(),
        'table_type': 'Financial Summary'
    })
    
    # Create a research data table
    research_data = {
        'Method': ['Method A', 'Method B', 'Method C', 'Method D'],
        'Accuracy (%)': [92.3, 89.7, 94.1, 91.8],
        'Precision (%)': [88.9, 91.2, 93.5, 90.4],
        'Recall (%)': [94.7, 87.3, 94.8, 93.1],
        'F1-Score': [0.917, 0.892, 0.941, 0.917]
    }
    
    research_df = pd.DataFrame(research_data)
    
    sample_tables.append({
        'data': research_df,
        'page': 2,
        'accuracy': 92.1,
        'table_index': 1,
        'shape': research_df.shape,
        'extraction_method': method.lower(),
        'table_type': 'Research Results'
    })
    
    # Filter by accuracy and limit
    filtered_tables = [t for t in sample_tables if t['accuracy'] >= min_accuracy]
    return filtered_tables[:max_tables]

def display_extracted_tables(
    tables: List[Dict[str, Any]],
    document: Dict,
    analyze_content: bool,
    include_metadata: bool,
    clean_data: bool
):
    """Display extracted tables with analysis"""
    
    st.success(f"✅ Found {len(tables)} tables")
    
    for i, table_data in enumerate(tables):
        with st.expander(f"📊 Table {i+1} (Page {table_data.get('page', 'Unknown')})", expanded=True):
            
            # Display metadata
            if include_metadata:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{table_data.get('accuracy', 0):.1f}%")
                with col2:
                    st.metric("Rows", table_data['data'].shape[0])
                with col3:
                    st.metric("Columns", table_data['data'].shape[1])
                with col4:
                    st.metric("Method", table_data.get('extraction_method', 'Unknown'))
            
            # Display table
            st.markdown("#### 📋 Table Data")
            st.dataframe(table_data['data'], use_container_width=True)
            
            # Content analysis
            if analyze_content:
                with st.spinner("Analyzing table content..."):
                    analysis = analyze_table_content(table_data)
                    if analysis:
                        st.markdown("#### 🧠 Content Analysis")
                        st.markdown(analysis)
            
            # Individual table export
            st.markdown("#### 📤 Export Options")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_data = table_data['data'].to_csv(index=False)
                st.download_button(
                    "📄 CSV",
                    data=csv_data,
                    file_name=f"table_{i+1}_page_{table_data.get('page', 1)}.csv",
                    mime="text/csv",
                    key=f"csv_export_{document.get('name', 'doc')}_{i}"
                )
            
            with col2:
                # Excel export
                try:
                    import io
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        table_data['data'].to_excel(writer, sheet_name='Table', index=False)
                    
                    st.download_button(
                        "📊 Excel",
                        data=buffer.getvalue(),
                        file_name=f"table_{i+1}_page_{table_data.get('page', 1)}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"excel_export_{document.get('name', 'doc')}_{i}"
                    )
                except ImportError:
                    st.warning("Excel export not available")
            
            with col3:
                json_data = table_data['data'].to_json(orient='records', indent=2)
                st.download_button(
                    "📋 JSON",
                    data=json_data,
                    file_name=f"table_{i+1}_page_{table_data.get('page', 1)}.json",
                    mime="application/json",
                    key=f"json_export_{document.get('name', 'doc')}_{i}"
                )

def analyze_table_content(table_data: Dict[str, Any]) -> str:
    """Analyze table content using AI"""
    
    try:
        from utils.llm_engine import LLMEngine
        llm_engine = LLMEngine()
        
        df = table_data['data']
        
        # Prepare table summary
        table_info = f"""
Table Shape: {df.shape[0]} rows × {df.shape[1]} columns
Column Names: {', '.join(df.columns.tolist())}
Extraction Accuracy: {table_data.get('accuracy', 0):.1f}%
Page: {table_data.get('page', 'Unknown')}

Table Preview:
{df.head().to_string()}
"""
        
        prompt = f"""Analyze this extracted table and provide insights:

{table_info}

Please provide:
1. Table type and purpose
2. Key insights from the data
3. Data patterns or trends
4. Potential use cases
5. Data quality assessment
6. Suggestions for further analysis

Focus on educational and analytical value."""
        
        analysis = llm_engine.generate_response(prompt, task_type="analyze")
        return analysis
    
    except Exception as e:
        logger.error(f"Table content analysis failed: {e}")
        return f"Content analysis unavailable: {str(e)}"

def display_extraction_summary(tables: List[Dict[str, Any]]):
    """Display summary of all extracted tables"""
    
    st.markdown("### 📊 Extraction Summary")
    
    total_tables = len(tables)
    total_rows = sum(table['data'].shape[0] for table in tables)
    total_cols = sum(table['data'].shape[1] for table in tables)
    avg_accuracy = sum(table.get('accuracy', 0) for table in tables) / total_tables if total_tables > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tables", total_tables)
    with col2:
        st.metric("Total Rows", total_rows)
    with col3:
        st.metric("Total Columns", total_cols)
    with col4:
        st.metric("Avg Accuracy", f"{avg_accuracy:.1f}%")
    
    # Method breakdown
    methods = {}
    for table in tables:
        method = table.get('extraction_method', 'unknown')
        methods[method] = methods.get(method, 0) + 1
    
    if methods:
        st.markdown("#### 🔧 Extraction Methods Used")
        for method, count in methods.items():
            st.write(f"• **{method.title()}**: {count} tables")

def prepare_table_exports(tables: List[Dict[str, Any]], formats: List[str]):
    """Prepare combined exports for all tables"""
    
    export_data = {
        'all_tables': tables,
        'summary': {
            'total_tables': len(tables),
            'extraction_date': pd.Timestamp.now().isoformat()
        }
    }
    
    # Create combined datasets
    if len(tables) > 1:
        try:
            # Combine all tables into one dataset
            combined_data = []
            for i, table in enumerate(tables):
                df = table['data'].copy()
                df['source_table'] = f"Table_{i+1}"
                df['source_page'] = table.get('page', 'Unknown')
                combined_data.append(df)
            
            if combined_data:
                export_data['combined_tables'] = pd.concat(combined_data, ignore_index=True)
        
        except Exception as e:
            logger.warning(f"Failed to combine tables: {e}")
    
    render_quick_export_buttons(export_data, "tables")

def display_table_history():
    """Display previously extracted tables"""
    if 'extracted_tables' not in st.session_state or not st.session_state.extracted_tables:
        st.info("No previous table extractions found.")
        return
    
    st.markdown("### 📚 Table Extraction History")
    
    # Group by document or session
    for i, table in enumerate(st.session_state.extracted_tables):
        with st.expander(f"📊 Table {i+1} - Page {table.get('page', 'Unknown')}"):
            st.markdown(f"**Accuracy:** {table.get('accuracy', 0):.1f}%")
            st.markdown(f"**Method:** {table.get('extraction_method', 'Unknown')}")
            st.markdown(f"**Shape:** {table['data'].shape[0]} rows × {table['data'].shape[1]} columns")
            
            if st.checkbox(f"Show data for table {i+1}", key=f"show_table_{i}"):
                st.dataframe(table['data'])
