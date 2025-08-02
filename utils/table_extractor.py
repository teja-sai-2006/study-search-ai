import logging
from typing import List, Dict, Any, Optional
import tempfile
import os
import pandas as pd

logger = logging.getLogger(__name__)

class TableExtractor:
    """Advanced table extraction from PDFs and documents"""
    
    def __init__(self):
        self.extractors = {
            'camelot': self._extract_with_camelot,
            'pdfplumber': self._extract_with_pdfplumber,
            'tabula': self._extract_with_tabula
        }
    
    def extract_tables_from_pdf(self, pdf_path: str, pages: str = "all", method: str = "auto") -> List[Dict[str, Any]]:
        """Extract tables from PDF using multiple methods"""
        if method == "auto":
            # Try multiple methods and return best results
            all_tables = []
            
            for extractor_name, extractor_func in self.extractors.items():
                try:
                    tables = extractor_func(pdf_path, pages)
                    for table in tables:
                        table['extraction_method'] = extractor_name
                    all_tables.extend(tables)
                except Exception as e:
                    logger.warning(f"{extractor_name} failed: {e}")
            
            return self._deduplicate_tables(all_tables)
        
        else:
            # Use specific method
            if method in self.extractors:
                try:
                    tables = self.extractors[method](pdf_path, pages)
                    for table in tables:
                        table['extraction_method'] = method
                    return tables
                except Exception as e:
                    logger.error(f"{method} extraction failed: {e}")
                    return []
            else:
                logger.error(f"Unknown extraction method: {method}")
                return []
    
    def _extract_with_camelot(self, pdf_path: str, pages: str = "all") -> List[Dict[str, Any]]:
        """Extract tables using Camelot"""
        try:
            import camelot
            
            tables = camelot.read_pdf(pdf_path, pages=pages, flavor='lattice')
            extracted_tables = []
            
            for i, table in enumerate(tables):
                if table.accuracy > 50:  # Only include tables with reasonable accuracy
                    df = table.df
                    extracted_tables.append({
                        'data': df,
                        'page': table.page,
                        'accuracy': table.accuracy,
                        'table_index': i,
                        'shape': df.shape,
                        'bbox': table._bbox if hasattr(table, '_bbox') else None
                    })
            
            return extracted_tables
        
        except ImportError:
            logger.error("Camelot not installed")
            return []
        except Exception as e:
            logger.error(f"Camelot extraction error: {e}")
            return []
    
    def _extract_with_pdfplumber(self, pdf_path: str, pages: str = "all") -> List[Dict[str, Any]]:
        """Extract tables using pdfplumber"""
        try:
            import pdfplumber
            
            extracted_tables = []
            
            with pdfplumber.open(pdf_path) as pdf:
                pages_to_process = self._parse_page_range(pages, len(pdf.pages))
                
                for page_num in pages_to_process:
                    page = pdf.pages[page_num - 1]  # pdfplumber uses 0-based indexing
                    tables = page.extract_tables()
                    
                    for i, table in enumerate(tables):
                        if table and len(table) > 1:  # Ensure table has data
                            # Convert to DataFrame
                            df = pd.DataFrame(table[1:], columns=table[0])
                            
                            extracted_tables.append({
                                'data': df,
                                'page': page_num,
                                'accuracy': 90,  # pdfplumber typically good accuracy
                                'table_index': i,
                                'shape': df.shape,
                                'bbox': None
                            })
            
            return extracted_tables
        
        except ImportError:
            logger.error("pdfplumber not installed")
            return []
        except Exception as e:
            logger.error(f"pdfplumber extraction error: {e}")
            return []
    
    def _extract_with_tabula(self, pdf_path: str, pages: str = "all") -> List[Dict[str, Any]]:
        """Extract tables using tabula-py"""
        try:
            import tabula
            
            extracted_tables = []
            
            # tabula returns list of DataFrames
            dfs = tabula.read_pdf(pdf_path, pages=pages, multiple_tables=True)
            
            for i, df in enumerate(dfs):
                if not df.empty:
                    extracted_tables.append({
                        'data': df,
                        'page': 1,  # tabula doesn't easily provide page numbers
                        'accuracy': 85,  # Estimate
                        'table_index': i,
                        'shape': df.shape,
                        'bbox': None
                    })
            
            return extracted_tables
        
        except ImportError:
            logger.error("tabula-py not installed (requires Java)")
            return []
        except Exception as e:
            logger.error(f"tabula extraction error: {e}")
            return []
    
    def _parse_page_range(self, pages: str, total_pages: int) -> List[int]:
        """Parse page range string into list of page numbers"""
        if pages == "all":
            return list(range(1, total_pages + 1))
        
        page_list = []
        for part in pages.split(','):
            part = part.strip()
            if '-' in part:
                start, end = part.split('-')
                page_list.extend(range(int(start), int(end) + 1))
            else:
                page_list.append(int(part))
        
        # Filter valid pages
        return [p for p in page_list if 1 <= p <= total_pages]
    
    def _deduplicate_tables(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate tables from different extraction methods"""
        if not tables:
            return []
        
        unique_tables = []
        
        for table in tables:
            is_duplicate = False
            df = table['data']
            
            for existing in unique_tables:
                existing_df = existing['data']
                
                # Check if tables are similar (same shape and similar content)
                if (df.shape == existing_df.shape and 
                    table.get('page') == existing.get('page')):
                    
                    # Simple similarity check
                    try:
                        similarity = self._calculate_table_similarity(df, existing_df)
                        if similarity > 0.8:
                            # Keep the one with higher accuracy
                            if table.get('accuracy', 0) > existing.get('accuracy', 0):
                                unique_tables.remove(existing)
                                unique_tables.append(table)
                            is_duplicate = True
                            break
                    except:
                        pass
            
            if not is_duplicate:
                unique_tables.append(table)
        
        return unique_tables
    
    def _calculate_table_similarity(self, df1: pd.DataFrame, df2: pd.DataFrame) -> float:
        """Calculate similarity between two DataFrames"""
        if df1.shape != df2.shape:
            return 0.0
        
        # Convert to string and compare
        str1 = df1.astype(str).values.flatten()
        str2 = df2.astype(str).values.flatten()
        
        matches = sum(1 for a, b in zip(str1, str2) if a == b)
        total = len(str1)
        
        return matches / total if total > 0 else 0.0
    
    def analyze_table(self, table_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze table structure and content"""
        df = table_data['data']
        
        analysis = {
            'rows': df.shape[0],
            'columns': df.shape[1],
            'column_names': df.columns.tolist(),
            'data_types': df.dtypes.to_dict(),
            'has_header': True,  # Assume tables have headers
            'numeric_columns': [],
            'text_columns': [],
            'summary': ""
        }
        
        # Analyze column types
        for col in df.columns:
            try:
                pd.to_numeric(df[col], errors='raise')
                analysis['numeric_columns'].append(col)
            except:
                analysis['text_columns'].append(col)
        
        # Generate summary using LLM
        try:
            from utils.llm_engine import LLMEngine
            llm = LLMEngine()
            
            table_preview = df.head().to_string()
            prompt = f"""Analyze this table and provide a brief summary:

{table_preview}

Table has {analysis['rows']} rows and {analysis['columns']} columns.
Numeric columns: {analysis['numeric_columns']}
Text columns: {analysis['text_columns']}

Please provide:
1. What type of data this table contains
2. Key insights or patterns
3. Potential use in educational context
"""
            
            analysis['summary'] = llm.generate_response(prompt, task_type="analyze")
        
        except Exception as e:
            logger.error(f"Table analysis failed: {e}")
            analysis['summary'] = "Basic table structure analysis completed"
        
        return analysis
    
    def export_table(self, table_data: Dict[str, Any], format: str = "csv") -> bytes:
        """Export table in specified format"""
        df = table_data['data']
        
        if format.lower() == "csv":
            return df.to_csv(index=False).encode('utf-8')
        
        elif format.lower() == "excel":
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Table', index=False)
            return buffer.getvalue()
        
        elif format.lower() == "json":
            return df.to_json(orient='records', indent=2).encode('utf-8')
        
        else:
            raise ValueError(f"Unsupported format: {format}")
