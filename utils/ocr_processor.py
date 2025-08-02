import logging
import tempfile
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

class OCRProcessor:
    """OCR processing using pytesseract with pdf2image for PDFs"""
    
    def __init__(self):
        self.ocr_available = self._check_ocr_availability()
    
    def _check_ocr_availability(self) -> bool:
        """Check if OCR dependencies are available"""
        try:
            import pytesseract
            import pdf2image
            from PIL import Image
            
            # Test tesseract
            pytesseract.get_tesseract_version()
            return True
            
        except ImportError as e:
            logger.warning(f"OCR dependencies not available: {e}")
            return False
        except Exception as e:
            logger.warning(f"Tesseract not properly configured: {e}")
            return False
    
    def extract_text_from_pdf(self, pdf_path: str, pages: List[int] = None) -> str:
        """Extract text from PDF using OCR"""
        if not self.ocr_available:
            return "❌ OCR not available. Please install pytesseract and pdf2image."
        
        try:
            from pdf2image import convert_from_path
            import pytesseract
            
            # Convert PDF to images
            if pages:
                images = convert_from_path(pdf_path, first_page=min(pages), last_page=max(pages))
            else:
                images = convert_from_path(pdf_path)
            
            text_content = ""
            
            for i, image in enumerate(images):
                try:
                    # Extract text from image
                    page_text = pytesseract.image_to_string(image, lang='eng')
                    if page_text.strip():
                        text_content += f"\n--- Page {i + 1} ---\n"
                        text_content += page_text + "\n"
                except Exception as e:
                    logger.warning(f"OCR failed for page {i + 1}: {e}")
            
            return text_content.strip() if text_content.strip() else "❌ No text extracted via OCR"
        
        except ImportError:
            return "❌ PDF to image conversion not available"
        except Exception as e:
            logger.error(f"PDF OCR failed: {e}")
            return f"❌ PDF OCR failed: {str(e)}"
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image using OCR"""
        if not self.ocr_available:
            return "❌ OCR not available. Please install pytesseract."
        
        try:
            import pytesseract
            from PIL import Image
            
            # Open and process image
            image = Image.open(image_path)
            
            # Extract text
            text = pytesseract.image_to_string(image, lang='eng')
            
            return text.strip() if text.strip() else "❌ No text found in image"
        
        except Exception as e:
            logger.error(f"Image OCR failed: {e}")
            return f"❌ Image OCR failed: {str(e)}"
    
    def extract_text_with_confidence(self, image_path: str) -> dict:
        """Extract text with confidence scores"""
        if not self.ocr_available:
            return {"text": "❌ OCR not available", "confidence": 0}
        
        try:
            import pytesseract
            from PIL import Image
            
            image = Image.open(image_path)
            
            # Get detailed data with confidence
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            # Filter out low confidence text
            texts = []
            confidences = []
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 30:  # Only include text with >30% confidence
                    text = data['text'][i].strip()
                    if text:
                        texts.append(text)
                        confidences.append(int(data['conf'][i]))
            
            combined_text = ' '.join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                "text": combined_text if combined_text else "❌ No confident text found",
                "confidence": avg_confidence,
                "word_count": len(texts)
            }
        
        except Exception as e:
            logger.error(f"Confidence OCR failed: {e}")
            return {"text": f"❌ OCR failed: {str(e)}", "confidence": 0}
    
    def preprocess_image_for_ocr(self, image_path: str) -> str:
        """Preprocess image to improve OCR accuracy"""
        if not self.ocr_available:
            return image_path
        
        try:
            from PIL import Image, ImageEnhance, ImageFilter
            import cv2
            import numpy as np
            
            # Open image
            image = Image.open(image_path)
            
            # Convert to grayscale if needed
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Apply slight gaussian blur to reduce noise
            image = image.filter(ImageFilter.GaussianBlur(0.5))
            
            # Save preprocessed image
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                image.save(tmp_file.name)
                return tmp_file.name
        
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return image_path
    
    def extract_text_from_multiple_images(self, image_paths: List[str]) -> str:
        """Extract text from multiple images"""
        all_text = ""
        
        for i, image_path in enumerate(image_paths):
            text = self.extract_text_from_image(image_path)
            if text and not text.startswith("❌"):
                all_text += f"\n--- Image {i + 1} ---\n"
                all_text += text + "\n"
        
        return all_text.strip() if all_text.strip() else "❌ No text extracted from images"
    
    def get_ocr_languages(self) -> List[str]:
        """Get available OCR languages"""
        if not self.ocr_available:
            return []
        
        try:
            import pytesseract
            langs = pytesseract.get_languages(config='')
            return langs
        except Exception as e:
            logger.error(f"Failed to get OCR languages: {e}")
            return ['eng']  # Default to English
    
    def extract_with_layout(self, image_path: str) -> dict:
        """Extract text while preserving layout information"""
        if not self.ocr_available:
            return {"text": "❌ OCR not available", "layout": []}
        
        try:
            import pytesseract
            from PIL import Image
            
            image = Image.open(image_path)
            
            # Get bounding box data
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            layout_info = []
            current_text = ""
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 30:
                    text = data['text'][i].strip()
                    if text:
                        layout_info.append({
                            'text': text,
                            'bbox': (data['left'][i], data['top'][i], 
                                   data['width'][i], data['height'][i]),
                            'confidence': int(data['conf'][i])
                        })
                        current_text += text + " "
            
            return {
                "text": current_text.strip(),
                "layout": layout_info
            }
        
        except Exception as e:
            logger.error(f"Layout OCR failed: {e}")
            return {"text": f"❌ Layout OCR failed: {str(e)}", "layout": []}
