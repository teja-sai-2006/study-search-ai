import os
import json
import logging
from typing import Dict, Any, Optional, List
import tempfile
import threading
from datetime import datetime

logger = logging.getLogger(__name__)

class OfflineModelManager:
    """Manages offline AI models for fallback functionality"""
    
    def __init__(self):
        self.models_dir = "models"
        self.config_file = "config/ai_config.json"
        self.available_models = {}
        self.loaded_models = {}
        self.model_lock = threading.Lock()
        self._load_model_config()
        self._check_model_availability()
    
    def _load_model_config(self):
        """Load model configuration from config file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                self.model_config = config.get('llm_config', {}).get('offline_models', {})
            else:
                self.model_config = self._get_default_model_config()
        
        except Exception as e:
            logger.error(f"Failed to load model config: {e}")
            self.model_config = self._get_default_model_config()
    
    def _get_default_model_config(self) -> Dict[str, Dict[str, Any]]:
        """Get default offline model configuration"""
        return {
            "dialo_gpt_small": {
                "name": "DialoGPT Small",
                "model_path": "microsoft/DialoGPT-small",
                "size_mb": 117,
                "capabilities": ["chat"],
                "best_for": "dialogue",
                "description": "Lightweight conversational model"
            },
            "flan_t5_base": {
                "name": "FLAN-T5 Base",
                "model_path": "google/flan-t5-base",
                "size_mb": 990,
                "capabilities": ["summarize", "generate"],
                "best_for": "summarization",
                "description": "Text-to-text generation model"
            },
            "granite_3_2b": {
                "name": "Granite 3.0 2B",
                "model_path": "ibm-granite/granite-3.0-2b-instruct",
                "size_mb": 2400,
                "capabilities": ["chat", "analyze"],
                "best_for": "qa",
                "description": "Instruction-tuned model for Q&A"
            },
            "mistral_7b_instruct": {
                "name": "Mistral 7B Instruct",
                "model_path": "mistralai/Mistral-7B-Instruct-v0.1",
                "size_mb": 7000,
                "capabilities": ["chat", "summarize", "analyze"],
                "best_for": "general",
                "description": "General purpose instruction model"
            },
            "phi_2": {
                "name": "Microsoft Phi-2",
                "model_path": "microsoft/phi-2",
                "size_mb": 2700,
                "capabilities": ["chat", "generate"],
                "best_for": "reasoning",
                "description": "Small model for reasoning tasks"
            }
        }
    
    def _check_model_availability(self):
        """Check which models are available locally"""
        try:
            # Import transformers to check availability
            import transformers
            from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
            
            for model_key, config in self.model_config.items():
                try:
                    model_path = config.get('model_path', '')
                    
                    # Check if model can be loaded
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_path,
                        cache_dir=self.models_dir,
                        local_files_only=False
                    )
                    
                    self.available_models[model_key] = {
                        'status': 'available',
                        'config': config,
                        'tokenizer_loaded': True,
                        'model_loaded': False
                    }
                    
                    logger.info(f"Model available: {config.get('name', model_key)}")
                
                except Exception as e:
                    logger.warning(f"Model not available: {config.get('name', model_key)} - {e}")
                    self.available_models[model_key] = {
                        'status': 'not_available',
                        'config': config,
                        'error': str(e)
                    }
        
        except ImportError:
            logger.error("Transformers library not available - offline models disabled")
            for model_key, config in self.model_config.items():
                self.available_models[model_key] = {
                    'status': 'dependencies_missing',
                    'config': config,
                    'error': 'transformers library not installed'
                }
    
    def get_available_models(self) -> Dict[str, str]:
        """Get status of all offline models"""
        status_map = {}
        
        for model_key, info in self.available_models.items():
            status = info.get('status', 'unknown')
            
            if status == 'available':
                status_map[model_key] = 'available'
            elif status == 'not_available':
                status_map[model_key] = 'not_downloaded'
            elif status == 'dependencies_missing':
                status_map[model_key] = 'dependencies_missing'
            else:
                status_map[model_key] = 'error'
        
        return status_map
    
    def load_model(self, model_key: str) -> bool:
        """Load a specific model into memory"""
        with self.model_lock:
            try:
                if model_key not in self.available_models:
                    logger.error(f"Model not configured: {model_key}")
                    return False
                
                if self.available_models[model_key]['status'] != 'available':
                    logger.error(f"Model not available: {model_key}")
                    return False
                
                if model_key in self.loaded_models:
                    logger.info(f"Model already loaded: {model_key}")
                    return True
                
                # Load model
                config = self.available_models[model_key]['config']
                model_path = config.get('model_path', '')
                
                logger.info(f"Loading model: {config.get('name', model_key)}")
                
                # Import required libraries
                from transformers import (
                    AutoTokenizer, 
                    AutoModelForCausalLM, 
                    AutoModelForSeq2SeqLM,
                    pipeline
                )
                import torch
                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    cache_dir=self.models_dir
                )
                
                # Determine model type and load accordingly
                if any(capability in config.get('capabilities', []) for capability in ['summarize']):
                    # Seq2Seq model (like T5)
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_path,
                        cache_dir=self.models_dir,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto" if torch.cuda.is_available() else None
                    )
                else:
                    # Causal LM model
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        cache_dir=self.models_dir,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto" if torch.cuda.is_available() else None
                    )
                
                # Store loaded model
                self.loaded_models[model_key] = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'config': config,
                    'loaded_at': datetime.now().isoformat(),
                    'device': str(model.device) if hasattr(model, 'device') else 'cpu'
                }
                
                # Update availability status
                self.available_models[model_key]['model_loaded'] = True
                
                logger.info(f"Model loaded successfully: {config.get('name', model_key)}")
                return True
            
            except Exception as e:
                logger.error(f"Failed to load model {model_key}: {e}")
                return False
    
    def unload_model(self, model_key: str) -> bool:
        """Unload a model from memory"""
        with self.model_lock:
            try:
                if model_key not in self.loaded_models:
                    return True
                
                # Clear model from memory
                if 'model' in self.loaded_models[model_key]:
                    del self.loaded_models[model_key]['model']
                
                if 'tokenizer' in self.loaded_models[model_key]:
                    del self.loaded_models[model_key]['tokenizer']
                
                del self.loaded_models[model_key]
                
                # Update availability status
                if model_key in self.available_models:
                    self.available_models[model_key]['model_loaded'] = False
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Clear CUDA cache if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                
                logger.info(f"Model unloaded: {model_key}")
                return True
            
            except Exception as e:
                logger.error(f"Failed to unload model {model_key}: {e}")
                return False
    
    def generate_response(self, model_key: str, prompt: str, context: str = "", task_type: str = "general") -> str:
        """Generate response using specified offline model"""
        try:
            # Load model if not already loaded
            if model_key not in self.loaded_models:
                if not self.load_model(model_key):
                    return f"❌ Failed to load model: {model_key}"
            
            model_info = self.loaded_models[model_key]
            model = model_info['model']
            tokenizer = model_info['tokenizer']
            config = model_info['config']
            
            # Prepare input based on task type and model capabilities
            input_text = self._prepare_input(prompt, context, task_type, config)
            
            # Generate response
            response = self._generate_with_model(model, tokenizer, input_text, config)
            
            return response
        
        except Exception as e:
            logger.error(f"Error generating response with {model_key}: {e}")
            return f"❌ Error generating response: {str(e)}"
    
    def _prepare_input(self, prompt: str, context: str, task_type: str, config: Dict[str, Any]) -> str:
        """Prepare input text based on model type and task"""
        model_name = config.get('name', '').lower()
        capabilities = config.get('capabilities', [])
        
        # Handle different model types
        if 'flan' in model_name or 't5' in model_name:
            # T5-style models prefer specific formats
            if task_type == "summarize":
                return f"summarize: {context}\n{prompt}"
            elif task_type == "chat":
                return f"answer the question: {prompt}\ncontext: {context}"
            else:
                return f"{prompt}\ncontext: {context}" if context else prompt
        
        elif 'dialo' in model_name:
            # DialogGPT style
            return prompt  # DialogGPT works best with simple prompts
        
        else:
            # General instruction format
            if context:
                return f"Context: {context}\n\nInstruction: {prompt}\n\nResponse:"
            else:
                return f"Instruction: {prompt}\n\nResponse:"
    
    def _generate_with_model(self, model, tokenizer, input_text: str, config: Dict[str, Any]) -> str:
        """Generate response using the model"""
        try:
            import torch
            
            # Tokenize input
            inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
            
            # Move to same device as model
            if hasattr(model, 'device'):
                inputs = inputs.to(model.device)
            
            # Generate
            with torch.no_grad():
                if hasattr(model, 'generate'):
                    # Use generate method for most models
                    outputs = model.generate(
                        inputs,
                        max_new_tokens=200,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id,
                        no_repeat_ngram_size=3
                    )
                    
                    # Extract only the generated part
                    if outputs.size(1) > inputs.size(1):
                        generated = outputs[0][inputs.size(1):]
                        response = tokenizer.decode(generated, skip_special_tokens=True)
                    else:
                        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                else:
                    # Fallback for models without generate method
                    outputs = model(inputs)
                    response = tokenizer.decode(outputs.logits.argmax(-1)[0], skip_special_tokens=True)
            
            # Clean up response
            response = response.strip()
            
            # Remove input text if it appears in output
            if input_text in response:
                response = response.replace(input_text, "").strip()
            
            return response if response else "I'm unable to generate a response to that."
        
        except Exception as e:
            logger.error(f"Error in model generation: {e}")
            return f"❌ Generation error: {str(e)}"
    
    def get_best_model_for_task(self, task_type: str) -> Optional[str]:
        """Get the best available model for a specific task"""
        task_preferences = {
            "chat": ["mistral_7b_instruct", "granite_3_2b", "phi_2", "dialo_gpt_small"],
            "summarize": ["flan_t5_base", "mistral_7b_instruct"],
            "analyze": ["granite_3_2b", "mistral_7b_instruct", "phi_2"],
            "generate": ["mistral_7b_instruct", "phi_2", "flan_t5_base"]
        }
        
        preferred_models = task_preferences.get(task_type, list(self.available_models.keys()))
        
        # Return first available model from preferences
        for model_key in preferred_models:
            if (model_key in self.available_models and 
                self.available_models[model_key]['status'] == 'available'):
                return model_key
        
        return None
    
    def get_model_info(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a model"""
        if model_key not in self.available_models:
            return None
        
        info = self.available_models[model_key].copy()
        
        # Add loading status
        info['is_loaded'] = model_key in self.loaded_models
        
        if model_key in self.loaded_models:
            loaded_info = self.loaded_models[model_key]
            info['loaded_at'] = loaded_info.get('loaded_at')
            info['device'] = loaded_info.get('device', 'unknown')
        
        return info
    
    def download_model(self, model_key: str) -> bool:
        """Download a model if not available locally"""
        try:
            if model_key not in self.model_config:
                logger.error(f"Model not configured: {model_key}")
                return False
            
            config = self.model_config[model_key]
            model_path = config.get('model_path', '')
            
            logger.info(f"Downloading model: {config.get('name', model_key)}")
            
            # Import required libraries
            from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
            
            # Download tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                cache_dir=self.models_dir
            )
            
            # Determine model type
            if any(capability in config.get('capabilities', []) for capability in ['summarize']):
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_path,
                    cache_dir=self.models_dir
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    cache_dir=self.models_dir
                )
            
            # Update availability
            self.available_models[model_key] = {
                'status': 'available',
                'config': config,
                'tokenizer_loaded': True,
                'model_loaded': False
            }
            
            logger.info(f"Model downloaded successfully: {config.get('name', model_key)}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to download model {model_key}: {e}")
            return False
    
    def cleanup_models(self) -> Dict[str, Any]:
        """Cleanup unused models and free memory"""
        cleanup_stats = {
            'unloaded_models': 0,
            'memory_freed': 0,
            'errors': []
        }
        
        try:
            # Get list of loaded models
            loaded_models = list(self.loaded_models.keys())
            
            for model_key in loaded_models:
                try:
                    if self.unload_model(model_key):
                        cleanup_stats['unloaded_models'] += 1
                except Exception as e:
                    cleanup_stats['errors'].append(f"Error unloading {model_key}: {str(e)}")
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    cleanup_stats['memory_freed'] = 1
            except ImportError:
                pass
            
            logger.info(f"Model cleanup completed: {cleanup_stats}")
            return cleanup_stats
        
        except Exception as e:
            logger.error(f"Error during model cleanup: {e}")
            cleanup_stats['errors'].append(str(e))
            return cleanup_stats
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        stats = {
            'loaded_models': len(self.loaded_models),
            'total_models': len(self.model_config),
            'available_models': len([m for m in self.available_models.values() if m['status'] == 'available']),
            'model_details': []
        }
        
        for model_key, info in self.loaded_models.items():
            config = info.get('config', {})
            stats['model_details'].append({
                'name': config.get('name', model_key),
                'size_mb': config.get('size_mb', 0),
                'device': info.get('device', 'unknown'),
                'loaded_at': info.get('loaded_at', '')
            })
        
        return stats
    
    def test_model_response(self, model_key: str) -> Dict[str, Any]:
        """Test a model with a simple prompt"""
        try:
            test_prompt = "Hello, how are you?"
            start_time = datetime.now()
            
            response = self.generate_response(model_key, test_prompt, task_type="chat")
            
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            return {
                'success': not response.startswith("❌"),
                'response': response,
                'response_time_seconds': response_time,
                'model_key': model_key,
                'test_prompt': test_prompt
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'model_key': model_key
            }
