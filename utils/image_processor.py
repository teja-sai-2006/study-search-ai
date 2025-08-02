import logging
from typing import Dict, Any, Optional, List
import tempfile
import os

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Image processing utilities for OCR and AI-powered captioning"""
    
    def __init__(self):
        self.captioning_model = None
        self._initialize_captioning_model()
    
    def _initialize_captioning_model(self):
        """Initialize image captioning model"""
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            import torch
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
            self.device = device
            self.captioning_model = True
        except ImportError:
            logger.warning("BLIP model dependencies not available")
            self.captioning_model = False
        except Exception as e:
            logger.error(f"Failed to initialize captioning model: {e}")
            self.captioning_model = False
    
    def process_image(self, image_file, extract_text: bool = True, generate_caption: bool = True) -> Dict[str, Any]:
        """Process image with OCR and captioning"""
        result = {
            'filename': image_file.name if hasattr(image_file, 'name') else 'image',
            'text': '',
            'caption': '',
            'description': '',
            'error': None
        }
        
        # Save image temporarily
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                if hasattr(image_file, 'getvalue'):
                    tmp_file.write(image_file.getvalue())
                else:
                    # Assume it's a file path
                    with open(image_file, 'rb') as f:
                        tmp_file.write(f.read())
                tmp_path = tmp_file.name
            
            # Extract text with OCR
            if extract_text:
                result['text'] = self.extract_text_from_image(tmp_path)
            
            # Generate caption
            if generate_caption:
                result['caption'] = self.generate_caption(tmp_path)
                result['description'] = self.generate_detailed_description(tmp_path)
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error processing image: {e}")
        
        finally:
            # Clean up
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        return result
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image using OCR"""
        try:
            from utils.ocr_processor import OCRProcessor
            ocr = OCRProcessor()
            return ocr.extract_text_from_image(image_path)
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return f"❌ OCR failed: {str(e)}"
    
    def generate_caption(self, image_path: str) -> str:
        """Generate caption for image using BLIP model"""
        if not self.captioning_model:
            return "❌ Image captioning model not available"
        
        try:
            from PIL import Image
            import torch
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            # Generate caption
            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=50, num_beams=4)
                caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            return caption
        
        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            return f"❌ Caption generation failed: {str(e)}"
    
    def generate_detailed_description(self, image_path: str) -> str:
        """Generate detailed description using AI model"""
        # Use the caption as base and enhance with LLM
        caption = self.generate_caption(image_path)
        
        if caption.startswith("❌"):
            return caption
        
        try:
            from utils.llm_engine import LLMEngine
            llm = LLMEngine()
            
            prompt = f"""Based on this image caption: "{caption}"
            
            Please provide a detailed description of what might be in this image, including:
            - Main objects or subjects
            - Setting or environment
            - Colors and composition
            - Any text that might be visible
            - Potential educational content or diagrams
            
            Keep the description informative and helpful for study purposes."""
            
            description = llm.generate_response(prompt, task_type="analyze")
            return description
        
        except Exception as e:
            logger.error(f"Detailed description generation failed: {e}")
            return f"Enhanced description not available. Basic caption: {caption}"
    
    def extract_images_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract images from PDF for processing"""
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(pdf_path)
            images = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Get image data
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            # Save image temporarily
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                                pix.save(tmp_file.name)
                                
                                # Process image
                                result = self.process_image(tmp_file.name)
                                result['page'] = page_num + 1
                                result['index'] = img_index
                                images.append(result)
                                
                                # Clean up
                                os.unlink(tmp_file.name)
                        
                        pix = None
                    
                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")
            
            doc.close()
            return images
        
        except ImportError:
            logger.error("PyMuPDF not available for image extraction")
            return []
        except Exception as e:
            logger.error(f"PDF image extraction failed: {e}")
            return []
    
    def analyze_chart_or_diagram(self, image_path: str) -> Dict[str, Any]:
        """Analyze charts, diagrams, and educational visuals"""
        result = self.process_image(image_path)
        
        # Enhanced analysis for educational content
        if result['caption'] and not result['caption'].startswith("❌"):
            try:
                from utils.llm_engine import LLMEngine
                llm = LLMEngine()
                
                prompt = f"""Analyze this educational image based on the caption: "{result['caption']}"
                
                If this appears to be a chart, diagram, or educational visual, please provide:
                1. Type of visual (chart, diagram, flowchart, etc.)
                2. Key information or data points
                3. Educational value and context
                4. How this might be used in studying
                5. Any formulas, concepts, or theories illustrated
                
                Format as a structured analysis."""
                
                analysis = llm.generate_response(prompt, result.get('text', ''), task_type="analyze")
                result['educational_analysis'] = analysis
            
            except Exception as e:
                logger.error(f"Educational analysis failed: {e}")
                result['educational_analysis'] = "Analysis not available"
        
        return result
