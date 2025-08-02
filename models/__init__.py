"""
StudyMate Models Package

This package contains AI model management functionality including
offline model handling, model switching, and fallback mechanisms.
"""

from .offline_models import OfflineModelManager

__all__ = ['OfflineModelManager']
