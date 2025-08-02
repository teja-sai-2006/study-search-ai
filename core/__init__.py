"""
StudyMate Core Package

This package contains core functionality for StudyMate including
session management, settings, and other foundational components.
"""

from .session_manager import SessionManager
from .settings_manager import SettingsManager

__all__ = ['SessionManager', 'SettingsManager']
