import os
import json
import logging
from typing import Dict, List, Optional, Any
import streamlit as st

logger = logging.getLogger(__name__)

class LLMEngine:
    """Enhanced LLM Engine with multi-model support and intelligent fallback"""
    
    def __init__(self):
        self.config = self._load_config()
        self.available_models = {}
        self.current_model = None
        self._initialize_models()
    
    def _load_config(self) -> Dict:
        """Load AI configuration"""
        try:
            with open("config/ai_config.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("AI config not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration if config file not found"""
        return {
            "llm_config": {
                "tier_1_online_models": {
                    "openai": {"name": "OpenAI GPT", "api_key_env": "OPENAI_API_KEY"},
                    "gemini": {"name": "Google Gemini", "api_key_env": "GEMINI_API_KEY"},
                    "anthropic": {"name": "Anthropic Claude", "api_key_env": "ANTHROPIC_API_KEY"}
                },
                "offline_models": {
                    "dialo_gpt_small": {"name": "DialoGPT Small", "model_path": "microsoft/DialoGPT-small"},
                    "flan_t5_base": {"name": "FLAN-T5 Base", "model_path": "google/flan-t5-base"},
                    "mistral_7b_instruct": {"name": "Mistral 7B", "model_path": "mistralai/Mistral-7B-Instruct-v0.1"}
                }
            }
        }
    
    def _initialize_models(self):
        """Initialize and check availability of all models"""
        # Check online models
        online_models = self.config.get('llm_config', {}).get('tier_1_online_models', {})
        for model_key, model_config in online_models.items():
            api_key_env = model_config.get('api_key_env')
            if api_key_env and os.getenv(api_key_env):
                self.available_models[model_key] = 'available'
            else:
                self.available_models[model_key] = 'not_configured'
        
        # Check offline models
        offline_models = self.config.get('llm_config', {}).get('offline_models', {})
        for model_key, model_config in offline_models.items():
            try:
                # Try to import transformers to check if offline models can be loaded
                import transformers
                self.available_models[model_key] = 'available'
            except ImportError:
                self.available_models[model_key] = 'not_downloaded'
    
    def get_available_models(self) -> Dict[str, str]:
        """Get status of all available models"""
        return self.available_models
    
    def select_best_model(self, task_type: str = "general") -> Optional[str]:
        """Intelligently select the best available model for a task"""
        module = st.session_state.get('selected_llm_module', 'auto_smart')
        
        if module == 'premium_apis':
            # Prefer online models
            online_models = self.config.get('llm_config', {}).get('tier_1_online_models', {})
            for model_key in online_models.keys():
                if self.available_models.get(model_key) == 'available':
                    return model_key
        
        elif module == 'offline_custom':
            # Use offline models only
            offline_models = self.config.get('llm_config', {}).get('offline_models', {})
            selected_offline = st.session_state.get('selected_offline_model')
            if selected_offline and self.available_models.get(selected_offline) == 'available':
                return selected_offline
            
            # Fallback to first available offline model
            for model_key in offline_models.keys():
                if self.available_models.get(model_key) == 'available':
                    return model_key
        
        else:  # auto_smart
            # Try online first, then offline
            all_models = list(self.config.get('llm_config', {}).get('tier_1_online_models', {}).keys())
            all_models.extend(list(self.config.get('llm_config', {}).get('offline_models', {}).keys()))
            
            for model_key in all_models:
                if self.available_models.get(model_key) == 'available':
                    return model_key
        
        return None
    
    def generate_response(self, prompt: str, context: str = "", task_type: str = "general") -> str:
        """Generate response using the selected model with fallback"""
        model = self.select_best_model(task_type)
        
        if not model:
            # Force use of offline models if available
            offline_models = self.config.get('llm_config', {}).get('offline_models', {})
            if offline_models:
                try:
                    return self._generate_offline_response(list(offline_models.keys())[0], prompt, context)
                except Exception as e:
                    logger.error(f"Offline model fallback failed: {e}")
            # Use simple fallback engine
            try:
                from utils.simple_llm_engine import get_simple_response
                return get_simple_response(prompt, context, task_type)
            except Exception as e:
                logger.error(f"Simple fallback failed: {e}")
                return "I apologize, but I'm currently unable to process your question due to model availability issues. Please try again or contact support if the problem persists."
        
        try:
            if model in self.config.get('llm_config', {}).get('tier_1_online_models', {}):
                return self._generate_online_response(model, prompt, context)
            else:
                return self._generate_offline_response(model, prompt, context)
        
        except Exception as e:
            logger.error(f"Error with model {model}: {e}")
            # Try fallback model
            fallback_model = self._get_fallback_model(model)
            if fallback_model:
                try:
                    if fallback_model in self.config.get('llm_config', {}).get('tier_1_online_models', {}):
                        return self._generate_online_response(fallback_model, prompt, context)
                    else:
                        return self._generate_offline_response(fallback_model, prompt, context)
                except Exception as fallback_error:
                    logger.error(f"Fallback model {fallback_model} also failed: {fallback_error}")
            
            return f"❌ Error generating response: {str(e)}"
    
    def _generate_online_response(self, model: str, prompt: str, context: str) -> str:
        """Generate response using online API models"""
        if model == 'openai':
            return self._call_openai_api(prompt, context)
        elif model == 'gemini':
            return self._call_gemini_api(prompt, context)
        elif model == 'anthropic':
            return self._call_anthropic_api(prompt, context)
        else:
            return f"❌ Unknown online model: {model}"
    
    def _generate_offline_response(self, model: str, prompt: str, context: str) -> str:
        """Generate response using offline models"""
        try:
            from models.offline_models import OfflineModelManager
            model_manager = OfflineModelManager()
            return model_manager.generate_response(model, prompt, context)
        except ImportError:
            return "❌ Offline models not available. Please install required dependencies."
        except Exception as e:
            return f"❌ Error with offline model: {str(e)}"
    
    def _call_openai_api(self, prompt: str, context: str) -> str:
        """Call OpenAI API"""
        try:
            import openai
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return "❌ OpenAI API key not configured"
            
            client = openai.OpenAI(api_key=api_key)
            
            messages = []
            if context:
                messages.append({"role": "system", "content": f"Context: {context}"})
            messages.append({"role": "user", "content": prompt})
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"❌ OpenAI API error: {str(e)}"
    
    def _call_gemini_api(self, prompt: str, context: str) -> str:
        """Call Gemini API"""
        try:
            import google.generativeai as genai
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return "❌ Gemini API key not configured"
            
            genai.configure(api_key=api_key)
            # Try newer model first, fallback to older version
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
            except:
                try:
                    model = genai.GenerativeModel('gemini-1.0-pro')
                except:
                    model = genai.GenerativeModel('gemini-pro')
            
            full_prompt = f"Context: {context}\n\nQuestion: {prompt}" if context else prompt
            response = model.generate_content(full_prompt)
            
            return response.text
        
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return f"❌ Gemini API error: {str(e)}"
    
    def _call_anthropic_api(self, prompt: str, context: str) -> str:
        """Call Anthropic Claude API"""
        try:
            import anthropic
            
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                return "❌ Anthropic API key not configured"
            
            client = anthropic.Anthropic(api_key=api_key)
            
            full_prompt = f"Context: {context}\n\nHuman: {prompt}\n\nAssistant:" if context else f"Human: {prompt}\n\nAssistant:"
            
            response = client.completions.create(
                model="claude-3-sonnet-20240229",
                prompt=full_prompt,
                max_tokens=1000
            )
            
            return response.completion
        
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return f"❌ Anthropic API error: {str(e)}"
    
    def _get_fallback_model(self, failed_model: str) -> Optional[str]:
        """Get fallback model when primary model fails"""
        online_models = list(self.config.get('llm_config', {}).get('tier_1_online_models', {}).keys())
        offline_models = list(self.config.get('llm_config', {}).get('offline_models', {}).keys())
        
        if failed_model in online_models:
            # Try other online models first, then offline
            for model in online_models:
                if model != failed_model and self.available_models.get(model) == 'available':
                    return model
            for model in offline_models:
                if self.available_models.get(model) == 'available':
                    return model
        else:
            # Failed model is offline, try other offline models
            for model in offline_models:
                if model != failed_model and self.available_models.get(model) == 'available':
                    return model
        
        return None
    
    def get_model_capabilities(self, model: str) -> List[str]:
        """Get capabilities of a specific model"""
        online_models = self.config.get('llm_config', {}).get('tier_1_online_models', {})
        offline_models = self.config.get('llm_config', {}).get('offline_models', {})
        
        if model in online_models:
            return online_models[model].get('capabilities', ['general'])
        elif model in offline_models:
            return offline_models[model].get('capabilities', ['general'])
        else:
            return ['general']
    
    def check_model_capacity(self, model: str, input_length: int) -> bool:
        """Check if model can handle the input length"""
        # Simple capacity check - can be enhanced
        if model in ['dialo_gpt_small', 'phi_2']:
            return input_length < 2000  # Small models
        elif model in ['flan_t5_base']:
            return input_length < 5000  # Medium models
        else:
            return input_length < 10000  # Large models
