"""
StudyMate UI Modes

Contains all the mode-specific UI implementations for StudyMate.
Each mode provides specialized functionality for different learning tasks.
"""

# Import all mode functions for easy access
from .chat_mode import render_chat_mode
from .summarize_mode import render_summarize_mode
from .customize_mode import render_customize_mode
from .topic_search_mode import render_topic_search_mode
from .image_mode import render_image_mode
from .advanced_tables_mode import render_advanced_tables_mode
from .web_search_mode import render_web_search_mode
from .study_planner_mode import render_study_planner_mode
from .flashcards_mode import render_flashcards_mode
from .study_progress_mode import render_study_progress_mode

__all__ = [
    'render_chat_mode',
    'render_summarize_mode', 
    'render_customize_mode',
    'render_topic_search_mode',
    'render_image_mode',
    'render_advanced_tables_mode',
    'render_web_search_mode',
    'render_study_planner_mode',
    'render_flashcards_mode',
    'render_study_progress_mode'
]
