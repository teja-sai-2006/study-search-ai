import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import streamlit as st

logger = logging.getLogger(__name__)

class SettingsManager:
    """Manages application settings, user preferences, and configuration"""
    
    def __init__(self):
        self.config_dir = "config"
        self.settings_file = os.path.join(self.config_dir, "user_settings.json")
        self.api_keys_file = os.path.join(self.config_dir, "api_keys.json")
        self._ensure_config_directory()
        self._load_settings()
    
    def _ensure_config_directory(self):
        """Ensure config directory exists"""
        try:
            os.makedirs(self.config_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create config directory: {e}")
    
    def _load_settings(self):
        """Load settings from file"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    self.settings = json.load(f)
            else:
                self.settings = self._get_default_settings()
                self._save_settings()
        
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            self.settings = self._get_default_settings()
    
    def _get_default_settings(self) -> Dict[str, Any]:
        """Get default application settings"""
        return {
            "ui_preferences": {
                "theme": "default",
                "language": "english",
                "default_summary_style": "Paragraph",
                "default_difficulty": "Intermediate",
                "auto_save": True,
                "enable_notifications": True,
                "show_tooltips": True,
                "compact_mode": False
            },
            "study_preferences": {
                "default_flashcard_types": ["Q&A", "Definitions", "Fill-in-the-blank"],
                "default_study_duration": 4,
                "default_daily_hours": 4,
                "default_break_interval": "30 minutes",
                "spaced_repetition_enabled": True,
                "progress_tracking_enabled": True,
                "goal_reminders_enabled": True
            },
            "ai_preferences": {
                "preferred_llm_module": "auto_smart",
                "default_model_timeout": 30,
                "enable_web_search": True,
                "web_search_sources": ["DuckDuckGo", "Wikipedia"],
                "max_context_length": 2000,
                "enable_model_fallback": True
            },
            "export_preferences": {
                "default_export_format": "PDF",
                "include_metadata": True,
                "include_timestamps": True,
                "auto_export_sessions": False,
                "export_location": "downloads"
            },
            "privacy_settings": {
                "store_chat_history": True,
                "store_search_history": True,
                "anonymous_analytics": False,
                "data_retention_days": 90
            },
            "advanced_settings": {
                "vector_db_size_mb": 500,
                "max_chat_history": 200,
                "ocr_quality": "Balanced",
                "enable_debug_mode": False,
                "auto_cleanup_enabled": True,
                "session_backup_enabled": True
            },
            "version": "1.0.0",
            "last_updated": datetime.now().isoformat()
        }
    
    def _save_settings(self):
        """Save settings to file"""
        try:
            self.settings["last_updated"] = datetime.now().isoformat()
            
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
            
            logger.info("Settings saved successfully")
        
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
    
    def get_setting(self, category: str, key: str, default: Any = None) -> Any:
        """Get a specific setting value"""
        try:
            return self.settings.get(category, {}).get(key, default)
        except Exception as e:
            logger.error(f"Error getting setting {category}.{key}: {e}")
            return default
    
    def set_setting(self, category: str, key: str, value: Any) -> bool:
        """Set a specific setting value"""
        try:
            if category not in self.settings:
                self.settings[category] = {}
            
            self.settings[category][key] = value
            self._save_settings()
            return True
        
        except Exception as e:
            logger.error(f"Error setting {category}.{key}: {e}")
            return False
    
    def get_ui_preferences(self) -> Dict[str, Any]:
        """Get UI preferences"""
        return self.settings.get("ui_preferences", {})
    
    def update_ui_preferences(self, preferences: Dict[str, Any]) -> bool:
        """Update UI preferences"""
        try:
            if "ui_preferences" not in self.settings:
                self.settings["ui_preferences"] = {}
            
            self.settings["ui_preferences"].update(preferences)
            self._save_settings()
            return True
        
        except Exception as e:
            logger.error(f"Failed to update UI preferences: {e}")
            return False
    
    def get_study_preferences(self) -> Dict[str, Any]:
        """Get study preferences"""
        return self.settings.get("study_preferences", {})
    
    def update_study_preferences(self, preferences: Dict[str, Any]) -> bool:
        """Update study preferences"""
        try:
            if "study_preferences" not in self.settings:
                self.settings["study_preferences"] = {}
            
            self.settings["study_preferences"].update(preferences)
            self._save_settings()
            return True
        
        except Exception as e:
            logger.error(f"Failed to update study preferences: {e}")
            return False
    
    def get_ai_preferences(self) -> Dict[str, Any]:
        """Get AI preferences"""
        return self.settings.get("ai_preferences", {})
    
    def update_ai_preferences(self, preferences: Dict[str, Any]) -> bool:
        """Update AI preferences"""
        try:
            if "ai_preferences" not in self.settings:
                self.settings["ai_preferences"] = {}
            
            self.settings["ai_preferences"].update(preferences)
            self._save_settings()
            return True
        
        except Exception as e:
            logger.error(f"Failed to update AI preferences: {e}")
            return False
    
    def save_api_keys(self, api_keys: Dict[str, str]) -> bool:
        """Save API keys securely"""
        try:
            # Filter out empty keys
            filtered_keys = {k: v for k, v in api_keys.items() if v and v.strip()}
            
            # Load existing keys
            existing_keys = self.load_api_keys()
            existing_keys.update(filtered_keys)
            
            # Save to file
            with open(self.api_keys_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "api_keys": existing_keys,
                    "last_updated": datetime.now().isoformat()
                }, f, indent=2)
            
            # Set environment variables
            for key, value in filtered_keys.items():
                env_var_name = self._get_env_var_name(key)
                os.environ[env_var_name] = value
            
            logger.info("API keys saved successfully")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save API keys: {e}")
            return False
    
    def load_api_keys(self) -> Dict[str, str]:
        """Load API keys from file"""
        try:
            if os.path.exists(self.api_keys_file):
                with open(self.api_keys_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                api_keys = data.get("api_keys", {})
                
                # Set environment variables
                for key, value in api_keys.items():
                    if value:
                        env_var_name = self._get_env_var_name(key)
                        os.environ[env_var_name] = value
                
                return api_keys
            
            return {}
        
        except Exception as e:
            logger.error(f"Failed to load API keys: {e}")
            return {}
    
    def _get_env_var_name(self, api_key_name: str) -> str:
        """Get environment variable name for API key"""
        mapping = {
            "OpenAI": "OPENAI_API_KEY",
            "Gemini": "GEMINI_API_KEY",
            "Anthropic": "ANTHROPIC_API_KEY",
            "IBM Watson": "IBM_WATSON_API_KEY"
        }
        return mapping.get(api_key_name, f"{api_key_name.upper().replace(' ', '_')}_API_KEY")
    
    def check_api_key_status(self) -> Dict[str, str]:
        """Check status of all API keys"""
        api_keys = self.load_api_keys()
        status = {}
        
        for key_name in ["OpenAI", "Gemini", "Anthropic"]:
            if key_name in api_keys and api_keys[key_name]:
                status[key_name] = "configured"
            else:
                env_var = self._get_env_var_name(key_name)
                if os.getenv(env_var):
                    status[key_name] = "configured_env"
                else:
                    status[key_name] = "not_configured"
        
        return status
    
    def delete_api_key(self, key_name: str) -> bool:
        """Delete a specific API key"""
        try:
            api_keys = self.load_api_keys()
            
            if key_name in api_keys:
                del api_keys[key_name]
                
                # Save updated keys
                with open(self.api_keys_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "api_keys": api_keys,
                        "last_updated": datetime.now().isoformat()
                    }, f, indent=2)
                
                # Remove from environment
                env_var_name = self._get_env_var_name(key_name)
                if env_var_name in os.environ:
                    del os.environ[env_var_name]
                
                logger.info(f"API key deleted: {key_name}")
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Failed to delete API key {key_name}: {e}")
            return False
    
    def export_settings(self) -> Optional[bytes]:
        """Export settings as downloadable file"""
        try:
            export_data = {
                "settings": self.settings,
                "export_date": datetime.now().isoformat(),
                "version": self.settings.get("version", "1.0.0")
            }
            
            # Don't include API keys in export for security
            export_json = json.dumps(export_data, indent=2, ensure_ascii=False)
            return export_json.encode('utf-8')
        
        except Exception as e:
            logger.error(f"Failed to export settings: {e}")
            return None
    
    def import_settings(self, settings_data: bytes) -> bool:
        """Import settings from uploaded file"""
        try:
            # Parse settings data
            settings_json = json.loads(settings_data.decode('utf-8'))
            
            # Validate structure
            if "settings" not in settings_json:
                logger.error("Invalid settings file format")
                return False
            
            imported_settings = settings_json["settings"]
            
            # Merge with current settings (preserve some user-specific settings)
            preserve_keys = ["api_keys", "privacy_settings"]
            
            for category, values in imported_settings.items():
                if category not in preserve_keys:
                    self.settings[category] = values
            
            # Update version and timestamp
            self.settings["version"] = imported_settings.get("version", "1.0.0")
            self.settings["imported_date"] = datetime.now().isoformat()
            
            self._save_settings()
            logger.info("Settings imported successfully")
            return True
        
        except Exception as e:
            logger.error(f"Failed to import settings: {e}")
            return False
    
    def reset_settings(self, category: Optional[str] = None) -> bool:
        """Reset settings to defaults"""
        try:
            defaults = self._get_default_settings()
            
            if category:
                # Reset specific category
                if category in defaults:
                    self.settings[category] = defaults[category]
                else:
                    logger.warning(f"Unknown settings category: {category}")
                    return False
            else:
                # Reset all settings
                self.settings = defaults
            
            self._save_settings()
            logger.info(f"Settings reset: {category or 'all'}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to reset settings: {e}")
            return False
    
    def get_theme_settings(self) -> Dict[str, Any]:
        """Get theme-related settings"""
        ui_prefs = self.get_ui_preferences()
        return {
            "theme": ui_prefs.get("theme", "default"),
            "compact_mode": ui_prefs.get("compact_mode", False),
            "show_tooltips": ui_prefs.get("show_tooltips", True)
        }
    
    def get_performance_settings(self) -> Dict[str, Any]:
        """Get performance-related settings"""
        advanced = self.settings.get("advanced_settings", {})
        ai_prefs = self.settings.get("ai_preferences", {})
        
        return {
            "vector_db_size_mb": advanced.get("vector_db_size_mb", 500),
            "max_chat_history": advanced.get("max_chat_history", 200),
            "model_timeout": ai_prefs.get("default_model_timeout", 30),
            "max_context_length": ai_prefs.get("max_context_length", 2000),
            "auto_cleanup_enabled": advanced.get("auto_cleanup_enabled", True)
        }
    
    def update_performance_settings(self, settings: Dict[str, Any]) -> bool:
        """Update performance settings"""
        try:
            # Update advanced settings
            advanced_updates = {
                "vector_db_size_mb": settings.get("vector_db_size_mb"),
                "max_chat_history": settings.get("max_chat_history"),
                "auto_cleanup_enabled": settings.get("auto_cleanup_enabled")
            }
            
            # Update AI preferences
            ai_updates = {
                "default_model_timeout": settings.get("model_timeout"),
                "max_context_length": settings.get("max_context_length")
            }
            
            # Filter out None values
            advanced_updates = {k: v for k, v in advanced_updates.items() if v is not None}
            ai_updates = {k: v for k, v in ai_updates.items() if v is not None}
            
            # Apply updates
            if advanced_updates:
                if "advanced_settings" not in self.settings:
                    self.settings["advanced_settings"] = {}
                self.settings["advanced_settings"].update(advanced_updates)
            
            if ai_updates:
                if "ai_preferences" not in self.settings:
                    self.settings["ai_preferences"] = {}
                self.settings["ai_preferences"].update(ai_updates)
            
            self._save_settings()
            return True
        
        except Exception as e:
            logger.error(f"Failed to update performance settings: {e}")
            return False
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all settings"""
        return self.settings.copy()
    
    def validate_settings(self) -> Dict[str, List[str]]:
        """Validate current settings and return any issues"""
        issues = {}
        
        try:
            # Validate UI preferences
            ui_prefs = self.settings.get("ui_preferences", {})
            ui_issues = []
            
            if ui_prefs.get("default_difficulty") not in ["Beginner", "Intermediate", "Advanced"]:
                ui_issues.append("Invalid default difficulty level")
            
            if ui_prefs.get("default_summary_style") not in ["Paragraph", "Bullet Points", "Table"]:
                ui_issues.append("Invalid default summary style")
            
            if ui_issues:
                issues["ui_preferences"] = ui_issues
            
            # Validate study preferences
            study_prefs = self.settings.get("study_preferences", {})
            study_issues = []
            
            daily_hours = study_prefs.get("default_daily_hours", 4)
            if not isinstance(daily_hours, (int, float)) or daily_hours < 1 or daily_hours > 24:
                study_issues.append("Invalid default daily hours (must be 1-24)")
            
            if study_issues:
                issues["study_preferences"] = study_issues
            
            # Validate advanced settings
            advanced = self.settings.get("advanced_settings", {})
            advanced_issues = []
            
            vector_db_size = advanced.get("vector_db_size_mb", 500)
            if not isinstance(vector_db_size, (int, float)) or vector_db_size < 100 or vector_db_size > 2000:
                advanced_issues.append("Invalid vector DB size (must be 100-2000 MB)")
            
            if advanced_issues:
                issues["advanced_settings"] = advanced_issues
        
        except Exception as e:
            logger.error(f"Error validating settings: {e}")
            issues["validation_error"] = [str(e)]
        
        return issues
    
    def backup_settings(self) -> Optional[str]:
        """Create backup of current settings"""
        try:
            backup_data = {
                "settings": self.settings,
                "backup_date": datetime.now().isoformat(),
                "version": self.settings.get("version", "1.0.0")
            }
            
            backup_filename = f"settings_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            backup_path = os.path.join(self.config_dir, backup_filename)
            
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Settings backed up to {backup_filename}")
            return backup_path
        
        except Exception as e:
            logger.error(f"Failed to backup settings: {e}")
            return None
