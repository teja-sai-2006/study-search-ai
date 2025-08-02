"""
Simple LLM Engine with reliable fallbacks for StudyMate
Provides basic text processing when advanced models aren't available
"""

import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class SimpleLLMEngine:
    """Simple text processing engine for basic AI functionality"""
    
    def __init__(self):
        self.templates = {
            'summarize': self._create_summary_templates(),
            'analyze': self._create_analysis_templates(),
            'chat': self._create_chat_templates(),
            'generate': self._create_generation_templates()
        }
    
    def _create_summary_templates(self) -> Dict[str, str]:
        return {
            'basic': "Here's a summary of the content:\n\n{content_preview}\n\nKey points:\n{key_points}",
            'advanced': "Detailed analysis:\n\n{content_preview}\n\nMain concepts:\n{key_points}\n\nImplications:\n{implications}"
        }
    
    def _create_analysis_templates(self) -> Dict[str, str]:
        return {
            'basic': "Analysis of '{topic}':\n\n{content_preview}\n\nMain findings:\n{findings}",
            'detailed': "Comprehensive analysis:\n\n{content_preview}\n\nKey insights:\n{insights}\n\nConclusions:\n{conclusions}"
        }
    
    def _create_chat_templates(self) -> Dict[str, str]:
        return {
            'response': "Based on the information provided:\n\n{context_summary}\n\nResponse:\n{simple_response}",
            'qa': "Question: {question}\n\nAnswer: {answer_content}"
        }
    
    def _create_generation_templates(self) -> Dict[str, str]:
        return {
            'basic': "Generated content:\n\n{generated_text}",
            'structured': "{title}\n\n{main_content}\n\nKey takeaways:\n{takeaways}"
        }
    
    def generate_response(self, prompt: str, context: str = "", task_type: str = "general") -> str:
        """Generate a basic response using simple text processing"""
        try:
            if task_type == "summarize":
                return self._generate_summary(prompt, context)
            elif task_type == "analyze":
                return self._generate_analysis(prompt, context)
            elif task_type == "chat":
                return self._generate_chat_response(prompt, context)
            elif task_type == "generate":
                return self._generate_content(prompt, context)
            else:
                return self._generate_general_response(prompt, context)
        
        except Exception as e:
            logger.error(f"Simple LLM fallback failed: {e}")
            return f"I can help you with that. Based on your request about: {prompt[:100]}...\n\nUnfortunately, I'm having technical difficulties right now. Please try uploading a document first, or contact support if this continues."
    
    def _generate_summary(self, prompt: str, context: str) -> str:
        """Generate a basic summary"""
        if not context:
            return "Please provide content to summarize."
        
        # Extract key information
        content_preview = context[:500] + ("..." if len(context) > 500 else "")
        
        # Simple key point extraction
        sentences = context.split('.')
        key_sentences = [s.strip() for s in sentences if len(s.strip()) > 30][:5]
        key_points = "\n".join([f"• {s}" for s in key_sentences])
        
        return self.templates['summarize']['basic'].format(
            content_preview=content_preview,
            key_points=key_points if key_points else "• Main content provided above"
        )
    
    def _generate_analysis(self, prompt: str, context: str) -> str:
        """Generate basic analysis"""
        if not context:
            return "Please provide content to analyze."
        
        content_preview = context[:300] + ("..." if len(context) > 300 else "")
        
        # Extract potential findings
        findings = [
            "Content contains relevant information about the topic",
            "Multiple concepts and ideas are discussed",
            "Further detailed analysis would provide additional insights"
        ]
        
        findings_text = "\n".join([f"• {f}" for f in findings])
        
        return self.templates['analyze']['basic'].format(
            topic=prompt[:50],
            content_preview=content_preview,
            findings=findings_text
        )
    
    def _generate_chat_response(self, prompt: str, context: str) -> str:
        """Generate basic chat response"""
        context_summary = context[:200] + ("..." if len(context) > 200 else "") if context else "No specific context provided"
        
        # Basic response patterns
        if "what" in prompt.lower():
            simple_response = "Based on the available information, this appears to be about the topic you've mentioned. Please provide more specific details for a more accurate response."
        elif "how" in prompt.lower():
            simple_response = "The process typically involves several steps. With more specific information, I could provide detailed guidance."
        elif "why" in prompt.lower():
            simple_response = "There are usually multiple factors involved. The context you've provided gives some insight into this question."
        else:
            simple_response = "I understand your question. Based on the information available, I can help guide you in the right direction."
        
        return self.templates['chat']['response'].format(
            context_summary=context_summary,
            simple_response=simple_response
        )
    
    def _generate_content(self, prompt: str, context: str) -> str:
        """Generate basic content"""
        title = f"Content about: {prompt[:50]}"
        
        if context:
            main_content = f"Based on the provided context:\n\n{context[:400]}..."
        else:
            main_content = "Content would be generated based on your specific requirements."
        
        takeaways = [
            "Review the content carefully",
            "Consider additional sources if needed", 
            "Apply information to your specific use case"
        ]
        
        takeaways_text = "\n".join([f"• {t}" for t in takeaways])
        
        return self.templates['generate']['structured'].format(
            title=title,
            main_content=main_content,
            takeaways=takeaways_text
        )
    
    def _generate_general_response(self, prompt: str, context: str) -> str:
        """Generate general response"""
        return f"""I understand you're asking about: {prompt[:100]}

Based on the information available:
{context[:300] + '...' if context and len(context) > 300 else context or 'No specific context provided'}

I'm ready to help you with this topic. For the best results, please:
• Provide specific documents or context
• Ask focused questions
• Use the specialized modes (Summarize, Chat, etc.) for better functionality

Would you like to try uploading a document or using one of the other features?"""

def get_simple_response(prompt: str, context: str = "", task_type: str = "general") -> str:
    """Quick function to get simple LLM response"""
    engine = SimpleLLMEngine()
    return engine.generate_response(prompt, context, task_type)