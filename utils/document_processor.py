import os
import tempfile
import zipfile
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import streamlit as st

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Enhanced document processor with multi-format support and OCR fallback"""
    
    def __init__(self):
        self.supported_formats = {
            'pdf': self._process_pdf,
            'docx': self._process_docx,
            'txt': self._process_txt,
            'md': self._process_markdown,
            'rtf': self._process_rtf,
            'xlsx': self._process_excel,
            'xls': self._process_excel,
            'csv': self._process_csv,
            'zip': self._process_zip,
            'png': self._process_image,
            'jpg': self._process_image,
            'jpeg': self._process_image
        }
    
    def process_document(self, uploaded_file) -> Dict[str, Any]:
        """Process uploaded document with format detection and OCR fallback"""
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                content = self.supported_formats[file_extension](tmp_path, uploaded_file.name)
                return {
                    'content': content,
                    'format': file_extension,
                    'name': uploaded_file.name,
                    'size': len(uploaded_file.getvalue())
                }
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        
        except Exception as e:
            logger.error(f"Error processing document {uploaded_file.name}: {e}")
            raise
    
    def process_zip_file(self, zip_file) -> List[Dict[str, Any]]:
        """Process ZIP file containing multiple documents"""
        processed_docs = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "uploaded.zip")
            
            # Save ZIP file
            with open(zip_path, 'wb') as f:
                f.write(zip_file.getvalue())
            
            # Extract ZIP file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Process each extracted file
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file == "uploaded.zip":
                        continue
                    
                    file_path = os.path.join(root, file)
                    file_extension = file.split('.')[-1].lower()
                    
                    if file_extension in self.supported_formats and file_extension != 'zip':
                        try:
                            content = self.supported_formats[file_extension](file_path, file)
                            processed_docs.append({
                                'content': content,
                                'format': file_extension,
                                'name': file,
                                'size': os.path.getsize(file_path)
                            })
                        except Exception as e:
                            logger.warning(f"Failed to process {file}: {e}")
        
        return processed_docs
    
    def _process_pdf(self, file_path: str, filename: str) -> str:
        """Process PDF with multiple extraction methods and OCR fallback"""
        try:
            # Try PyMuPDF first
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            text = ""
            
            for page in doc:
                text += page.get_text()
            doc.close()
            
            if text.strip():
                return text
            else:
                # Fallback to OCR
                return self._ocr_fallback(file_path)
        
        except ImportError:
            try:
                # Try pdfplumber
                import pdfplumber
                text = ""
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                
                if text.strip():
                    return text
                else:
                    return self._ocr_fallback(file_path)
            
            except ImportError:
                # OCR fallback
                return self._ocr_fallback(file_path)
    
    def _process_docx(self, file_path: str, filename: str) -> str:
        """Process DOCX files"""
        try:
            from docx import Document
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except ImportError:
            raise ImportError("python-docx not installed. Cannot process DOCX files.")
    
    def _process_txt(self, file_path: str, filename: str) -> str:
        """Process text files"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    def _process_markdown(self, file_path: str, filename: str) -> str:
        """Process Markdown files"""
        return self._process_txt(file_path, filename)
    
    def _process_rtf(self, file_path: str, filename: str) -> str:
        """Process RTF files"""
        try:
            from striprtf.striprtf import rtf_to_text
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                rtf_content = f.read()
            return rtf_to_text(rtf_content)
        except ImportError:
            # Fallback: read as plain text (will have RTF formatting)
            return self._process_txt(file_path, filename)
    
    def _process_excel(self, file_path: str, filename: str) -> str:
        """Process Excel files"""
        try:
            import pandas as pd
            
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            content = ""
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                content += f"Sheet: {sheet_name}\n"
                content += df.to_string(index=False) + "\n\n"
            
            return content
        except ImportError:
            raise ImportError("pandas and openpyxl not installed. Cannot process Excel files.")
    
    def _process_csv(self, file_path: str, filename: str) -> str:
        """Process CSV files"""
        try:
            import pandas as pd
            df = pd.read_csv(file_path)
            return df.to_string(index=False)
        except ImportError:
            # Fallback: read as text
            return self._process_txt(file_path, filename)
    
    def _process_zip(self, file_path: str, filename: str) -> str:
        """Process ZIP files (this should not be called directly)"""
        return "ZIP file processed separately"
    
    def _process_image(self, file_path: str, filename: str) -> str:
        """Process image files with OCR"""
        return self._ocr_image(file_path)
    
    def _ocr_fallback(self, file_path: str) -> str:
        """OCR fallback for PDFs that couldn't be parsed"""
        try:
            from utils.ocr_processor import OCRProcessor
            ocr = OCRProcessor()
            return ocr.extract_text_from_pdf(file_path)
        except Exception as e:
            logger.error(f"OCR fallback failed: {e}")
            return f"❌ Could not extract text from document. OCR failed: {str(e)}"
    
    def _ocr_image(self, file_path: str) -> str:
        """Extract text from image using OCR"""
        try:
            from utils.ocr_processor import OCRProcessor
            ocr = OCRProcessor()
            return ocr.extract_text_from_image(file_path)
        except Exception as e:
            logger.error(f"Image OCR failed: {e}")
            return f"❌ Could not extract text from image: {str(e)}"
    
    def get_document_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from document"""
        metadata = {
            'size': os.path.getsize(file_path),
            'extension': Path(file_path).suffix.lower(),
            'name': Path(file_path).name
        }
        
        try:
            # Try to get PDF metadata
            import fitz
            if metadata['extension'] == '.pdf':
                doc = fitz.open(file_path)
                pdf_metadata = doc.metadata
                metadata.update({
                    'title': pdf_metadata.get('title', ''),
                    'author': pdf_metadata.get('author', ''),
                    'pages': doc.page_count
                })
                doc.close()
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Could not extract metadata: {e}")
        
        return metadata
    
    def chunk_document(self, content: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split document into chunks for vector processing"""
        if len(content) <= chunk_size:
            return [content]
        
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + chunk_size
            
            # Try to break at sentence or paragraph
            if end < len(content):
                # Look for sentence end
                for i in range(end, max(start + chunk_size // 2, end - 200), -1):
                    if content[i] in '.!?\n':
                        end = i + 1
                        break
            
            chunk = content[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks
