import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class SimpleLLMEngine:
    """Simplified LLM Engine that provides basic functionality without AI dependencies"""
    
    def __init__(self):
        self.available_models = []
        self.current_model = None
        self.offline_available = False
        
    def check_model_availability(self) -> Dict[str, str]:
        """Check availability of AI models"""
        status = {
            "openai": "not_configured",
            "gemini": "not_configured", 
            "anthropic": "not_configured",
            "offline": "dependencies_missing"
        }
        
        # Check environment variables for API keys
        import os
        if os.getenv("OPENAI_API_KEY"):
            status["openai"] = "available"
        if os.getenv("GEMINI_API_KEY"):
            status["gemini"] = "available"
        if os.getenv("ANTHROPIC_API_KEY"):
            status["anthropic"] = "available"
            
        return status
    
    def generate_response(self, prompt: str, model: str = "fallback", context: str = "", **kwargs) -> str:
        """Generate a response using available models or fallback"""
        
        # Check for API keys and try online models first
        status = self.check_model_availability()
        
        # For now, return a helpful message explaining the situation
        if any(s == "available" for s in status.values()):
            return self._generate_with_api(prompt, model, context, **kwargs)
        else:
            return self._generate_fallback_response(prompt, context)
    
    def _generate_with_api(self, prompt: str, model: str, context: str, **kwargs) -> str:
        """Generate response using available API"""
        # This would contain actual API calls when dependencies are available
        return ("I can see you have API keys configured, but the AI libraries aren't fully installed yet. "
                "Please provide your API keys through the settings panel, and I'll help you get the AI features working.")
    
    def _generate_fallback_response(self, prompt: str, context: str) -> str:
        """Provide fallback response when no AI models are available"""
        return (f"StudyMate is ready to help! To enable AI features, please:\n\n"
                f"1. Add your API keys in the Settings panel\n"
                f"2. Choose from OpenAI, Google Gemini, or Anthropic Claude\n"
                f"3. Or wait while offline models are being prepared\n\n"
                f"Your question: '{prompt[:100]}...' will be answered once AI is configured.")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models"""
        return {
            "current_model": self.current_model or "none",
            "available_models": self.available_models,
            "status": self.check_model_availability()
        }
    
    def set_model(self, model_name: str) -> bool:
        """Set the current model"""
        self.current_model = model_name
        return True
        
    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        status = self.check_model_availability()
        available = []
        
        for model, state in status.items():
            if state == "available":
                available.append(model)
                
        return available if available else ["fallback"]