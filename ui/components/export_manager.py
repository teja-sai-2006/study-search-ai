import streamlit as st
import json
import pandas as pd
from typing import Dict, List, Any, Optional
import io
import zipfile
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ExportManager:
    """Manages export functionality for various StudyMate content"""
    
    def __init__(self):
        self.supported_formats = {
            'pdf': self._export_to_pdf,
            'markdown': self._export_to_markdown,
            'json': self._export_to_json,
            'csv': self._export_to_csv,
            'excel': self._export_to_excel,
            'zip': self._export_to_zip
        }
    
    def render_export_interface(self, content_types: List[str], selected_content: Dict[str, Any]):
        """Render export interface"""
        st.markdown("### 📤 Export Manager")
        
        # Content selection
        col1, col2 = st.columns(2)
        
        with col1:
            export_items = st.multiselect(
                "Select items to export:",
                content_types,
                default=[],
                help="Choose which content to include in the export"
            )
        
        with col2:
            export_format = st.selectbox(
                "Export format:",
                ["PDF", "Markdown", "JSON", "CSV", "Excel", "ZIP"],
                help="Select the output format for your export"
            )
        
        # Export options
        with st.expander("🔧 Export Options"):
            include_metadata = st.checkbox("Include metadata", value=True)
            include_timestamps = st.checkbox("Include timestamps", value=True)
            compress_output = st.checkbox("Compress output", value=False)
        
        # Preview selected content
        if export_items:
            st.markdown("#### 👁️ Export Preview")
            for item in export_items:
                if item in selected_content:
                    with st.expander(f"📄 {item}"):
                        content = selected_content[item]
                        if isinstance(content, str):
                            st.text_area("Content", content[:500] + "..." if len(content) > 500 else content, height=100, disabled=True)
                        elif isinstance(content, list):
                            st.write(f"Items: {len(content)}")
                        elif isinstance(content, dict):
                            st.json(content)
        
        # Export button
        if st.button("📤 Generate Export", type="primary"):
            if export_items:
                export_data = self._prepare_export_data(
                    export_items, 
                    selected_content, 
                    include_metadata, 
                    include_timestamps
                )
                
                try:
                    exported_content = self._export_content(export_data, export_format.lower())
                    
                    if exported_content:
                        filename = f"studymate_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format.lower()}"
                        
                        st.download_button(
                            "📥 Download Export",
                            data=exported_content,
                            file_name=filename,
                            mime=self._get_mime_type(export_format.lower())
                        )
                        
                        st.success(f"✅ Export ready! Click to download {filename}")
                    else:
                        st.error("❌ Export generation failed")
                
                except Exception as e:
                    st.error(f"❌ Export error: {str(e)}")
                    logger.error(f"Export failed: {e}")
            else:
                st.warning("⚠️ Please select items to export")
    
    def _prepare_export_data(self, items: List[str], content: Dict[str, Any], 
                           include_metadata: bool, include_timestamps: bool) -> Dict[str, Any]:
        """Prepare data for export"""
        export_data = {
            "export_info": {
                "generated_at": datetime.now().isoformat() if include_timestamps else None,
                "studymate_version": "1.0.0",
                "items_exported": items
            },
            "content": {}
        }
        
        for item in items:
            if item in content:
                item_data = content[item]
                
                if include_metadata:
                    export_data["content"][item] = {
                        "data": item_data,
                        "metadata": {
                            "type": type(item_data).__name__,
                            "size": len(str(item_data)) if item_data else 0,
                            "exported_at": datetime.now().isoformat() if include_timestamps else None
                        }
                    }
                else:
                    export_data["content"][item] = item_data
        
        return export_data
    
    def _export_content(self, data: Dict[str, Any], format: str) -> bytes:
        """Export content in specified format"""
        if format in self.supported_formats:
            return self.supported_formats[format](data)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_to_pdf(self, data: Dict[str, Any]) -> bytes:
        """Export to PDF format"""
        try:
            from fpdf import FPDF
            
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            
            # Add title
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "StudyMate Export", ln=True, align="C")
            pdf.ln(10)
            
            # Add export info
            if "export_info" in data:
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, "Export Information", ln=True)
                pdf.set_font("Arial", size=10)
                
                for key, value in data["export_info"].items():
                    if value:
                        pdf.cell(0, 8, f"{key}: {value}", ln=True)
                pdf.ln(5)
            
            # Add content
            for item_name, item_content in data.get("content", {}).items():
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, f"{item_name.title()}", ln=True)
                pdf.set_font("Arial", size=10)
                
                content_text = str(item_content)
                if len(content_text) > 1000:
                    content_text = content_text[:1000] + "..."
                
                # Split text into lines that fit
                lines = content_text.split('\n')
                for line in lines:
                    if len(line) > 80:
                        # Split long lines
                        words = line.split(' ')
                        current_line = ""
                        for word in words:
                            if len(current_line + word) < 80:
                                current_line += word + " "
                            else:
                                if current_line:
                                    pdf.cell(0, 6, current_line.strip(), ln=True)
                                current_line = word + " "
                        if current_line:
                            pdf.cell(0, 6, current_line.strip(), ln=True)
                    else:
                        pdf.cell(0, 6, line, ln=True)
                
                pdf.ln(5)
            
            return pdf.output(dest='S').encode('latin-1')
        
        except ImportError:
            # Fallback to text-based PDF
            content_text = self._convert_to_text(data)
            return content_text.encode('utf-8')
    
    def _export_to_markdown(self, data: Dict[str, Any]) -> bytes:
        """Export to Markdown format"""
        md_content = "# StudyMate Export\n\n"
        
        # Add export info
        if "export_info" in data:
            md_content += "## Export Information\n\n"
            for key, value in data["export_info"].items():
                if value:
                    md_content += f"- **{key}**: {value}\n"
            md_content += "\n"
        
        # Add content
        for item_name, item_content in data.get("content", {}).items():
            md_content += f"## {item_name.title()}\n\n"
            
            if isinstance(item_content, dict) and "data" in item_content:
                content = item_content["data"]
            else:
                content = item_content
            
            if isinstance(content, list):
                for i, item in enumerate(content):
                    md_content += f"{i+1}. {item}\n"
            elif isinstance(content, dict):
                md_content += "```json\n"
                md_content += json.dumps(content, indent=2)
                md_content += "\n```\n"
            else:
                md_content += f"{content}\n"
            
            md_content += "\n"
        
        return md_content.encode('utf-8')
    
    def _export_to_json(self, data: Dict[str, Any]) -> bytes:
        """Export to JSON format"""
        return json.dumps(data, indent=2, ensure_ascii=False).encode('utf-8')
    
    def _export_to_csv(self, data: Dict[str, Any]) -> bytes:
        """Export to CSV format"""
        # Convert data to tabular format
        rows = []
        
        for item_name, item_content in data.get("content", {}).items():
            if isinstance(item_content, dict) and "data" in item_content:
                content = item_content["data"]
            else:
                content = item_content
            
            if isinstance(content, list):
                for i, item in enumerate(content):
                    rows.append({
                        "Item": item_name,
                        "Index": i,
                        "Content": str(item)
                    })
            else:
                rows.append({
                    "Item": item_name,
                    "Index": 0,
                    "Content": str(content)
                })
        
        if rows:
            df = pd.DataFrame(rows)
            return df.to_csv(index=False).encode('utf-8')
        else:
            return "No data to export".encode('utf-8')
    
    def _export_to_excel(self, data: Dict[str, Any]) -> bytes:
        """Export to Excel format"""
        buffer = io.BytesIO()
        
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            # Export info sheet
            if "export_info" in data:
                info_df = pd.DataFrame(list(data["export_info"].items()), 
                                     columns=["Property", "Value"])
                info_df.to_excel(writer, sheet_name="Export Info", index=False)
            
            # Content sheets
            for item_name, item_content in data.get("content", {}).items():
                if isinstance(item_content, dict) and "data" in item_content:
                    content = item_content["data"]
                else:
                    content = item_content
                
                try:
                    if isinstance(content, list):
                        if content and isinstance(content[0], dict):
                            # List of dictionaries -> DataFrame
                            df = pd.DataFrame(content)
                        else:
                            # Simple list
                            df = pd.DataFrame({"Items": content})
                    elif isinstance(content, dict):
                        # Dictionary -> DataFrame
                        df = pd.DataFrame([content])
                    else:
                        # String or other -> Single cell
                        df = pd.DataFrame({"Content": [str(content)]})
                    
                    # Clean sheet name
                    sheet_name = item_name.replace(" ", "_")[:31]
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                except Exception as e:
                    logger.warning(f"Failed to export {item_name} to Excel: {e}")
        
        return buffer.getvalue()
    
    def _export_to_zip(self, data: Dict[str, Any]) -> bytes:
        """Export to ZIP archive with multiple formats"""
        buffer = io.BytesIO()
        
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add JSON version
            json_content = self._export_to_json(data)
            zip_file.writestr("export.json", json_content)
            
            # Add Markdown version
            md_content = self._export_to_markdown(data)
            zip_file.writestr("export.md", md_content)
            
            # Add CSV version
            csv_content = self._export_to_csv(data)
            zip_file.writestr("export.csv", csv_content)
            
            # Add individual files for each content item
            for item_name, item_content in data.get("content", {}).items():
                filename = f"{item_name.replace(' ', '_')}.txt"
                content_str = str(item_content)
                zip_file.writestr(filename, content_str.encode('utf-8'))
        
        return buffer.getvalue()
    
    def _convert_to_text(self, data: Dict[str, Any]) -> str:
        """Convert data to plain text"""
        text = "StudyMate Export\n" + "=" * 50 + "\n\n"
        
        if "export_info" in data:
            text += "Export Information:\n"
            for key, value in data["export_info"].items():
                if value:
                    text += f"  {key}: {value}\n"
            text += "\n"
        
        for item_name, item_content in data.get("content", {}).items():
            text += f"{item_name.title()}:\n"
            text += "-" * len(item_name) + "\n"
            text += str(item_content) + "\n\n"
        
        return text
    
    def _get_mime_type(self, format: str) -> str:
        """Get MIME type for format"""
        mime_types = {
            'pdf': 'application/pdf',
            'markdown': 'text/markdown',
            'json': 'application/json',
            'csv': 'text/csv',
            'excel': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'zip': 'application/zip'
        }
        return mime_types.get(format, 'application/octet-stream')

def render_quick_export_buttons(content: Dict[str, Any], prefix: str = "quick"):
    """Render quick export buttons for common formats"""
    if not content:
        return
    
    st.markdown("#### 📤 Quick Export")
    col1, col2, col3, col4 = st.columns(4)
    
    export_manager = ExportManager()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with col1:
        if st.button("📄 PDF", key=f"{prefix}_pdf"):
            try:
                pdf_content = export_manager._export_to_pdf({"content": content})
                st.download_button(
                    "Download PDF",
                    data=pdf_content,
                    file_name=f"studymate_{prefix}_{timestamp}.pdf",
                    mime="application/pdf",
                    key=f"{prefix}_pdf_download"
                )
            except Exception as e:
                st.error(f"PDF export failed: {e}")
    
    with col2:
        if st.button("📝 Markdown", key=f"{prefix}_md"):
            try:
                md_content = export_manager._export_to_markdown({"content": content})
                st.download_button(
                    "Download MD",
                    data=md_content,
                    file_name=f"studymate_{prefix}_{timestamp}.md",
                    mime="text/markdown",
                    key=f"{prefix}_md_download"
                )
            except Exception as e:
                st.error(f"Markdown export failed: {e}")
    
    with col3:
        if st.button("📊 JSON", key=f"{prefix}_json"):
            try:
                json_content = export_manager._export_to_json({"content": content})
                st.download_button(
                    "Download JSON",
                    data=json_content,
                    file_name=f"studymate_{prefix}_{timestamp}.json",
                    mime="application/json",
                    key=f"{prefix}_json_download"
                )
            except Exception as e:
                st.error(f"JSON export failed: {e}")
    
    with col4:
        if st.button("📦 ZIP", key=f"{prefix}_zip"):
            try:
                zip_content = export_manager._export_to_zip({"content": content})
                st.download_button(
                    "Download ZIP",
                    data=zip_content,
                    file_name=f"studymate_{prefix}_{timestamp}.zip",
                    mime="application/zip",
                    key=f"{prefix}_zip_download"
                )
            except Exception as e:
                st.error(f"ZIP export failed: {e}")
