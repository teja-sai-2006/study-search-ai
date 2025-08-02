import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import streamlit as st
import tempfile
import shutil

logger = logging.getLogger(__name__)

class SessionManager:
    """Manages user sessions, data persistence, and state management"""
    
    def __init__(self):
        self.sessions_dir = "sessions"
        self.current_session_id = None
        self._ensure_sessions_directory()
    
    def _ensure_sessions_directory(self):
        """Ensure sessions directory exists"""
        try:
            os.makedirs(self.sessions_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create sessions directory: {e}")
    
    def save_session(self, session_name: str, session_state: Dict[str, Any]) -> bool:
        """Save current session state to file"""
        try:
            # Clean session name for filename
            safe_name = "".join(c for c in session_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_name = safe_name.replace(' ', '_')
            
            session_id = f"{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            session_file = os.path.join(self.sessions_dir, f"{session_id}.json")
            
            # Prepare session data
            session_data = {
                'session_id': session_id,
                'name': session_name,
                'created_date': datetime.now().isoformat(),
                'state': self._serialize_session_state(session_state)
            }
            
            # Save to file
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Session saved: {session_name} -> {session_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save session '{session_name}': {e}")
            return False
    
    def load_session(self, session_name: str) -> Optional[Dict[str, Any]]:
        """Load session state from file"""
        try:
            session_file = self._find_session_file(session_name)
            if not session_file:
                logger.warning(f"Session not found: {session_name}")
                return None
            
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            # Deserialize and return state
            state = self._deserialize_session_state(session_data.get('state', {}))
            
            logger.info(f"Session loaded: {session_name}")
            return state
        
        except Exception as e:
            logger.error(f"Failed to load session '{session_name}': {e}")
            return None
    
    def list_sessions(self) -> List[str]:
        """List all available sessions"""
        try:
            sessions = []
            
            if not os.path.exists(self.sessions_dir):
                return sessions
            
            for filename in os.listdir(self.sessions_dir):
                if filename.endswith('.json'):
                    try:
                        filepath = os.path.join(self.sessions_dir, filename)
                        with open(filepath, 'r', encoding='utf-8') as f:
                            session_data = json.load(f)
                        
                        session_name = session_data.get('name', filename[:-5])
                        created_date = session_data.get('created_date', '')
                        
                        sessions.append({
                            'name': session_name,
                            'created_date': created_date,
                            'file': filename
                        })
                    
                    except Exception as e:
                        logger.warning(f"Failed to read session file {filename}: {e}")
            
            # Sort by creation date (newest first)
            sessions.sort(key=lambda x: x.get('created_date', ''), reverse=True)
            
            return [s['name'] for s in sessions]
        
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []
    
    def delete_session(self, session_name: str) -> bool:
        """Delete a session"""
        try:
            session_file = self._find_session_file(session_name)
            if not session_file:
                return False
            
            os.remove(session_file)
            logger.info(f"Session deleted: {session_name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete session '{session_name}': {e}")
            return False
    
    def export_session(self, session_name: str) -> Optional[bytes]:
        """Export session as downloadable file"""
        try:
            session_file = self._find_session_file(session_name)
            if not session_file:
                return None
            
            with open(session_file, 'rb') as f:
                return f.read()
        
        except Exception as e:
            logger.error(f"Failed to export session '{session_name}': {e}")
            return None
    
    def import_session(self, session_data: bytes, session_name: str = None) -> bool:
        """Import session from uploaded file"""
        try:
            # Parse session data
            session_json = json.loads(session_data.decode('utf-8'))
            
            # Use provided name or extract from data
            name = session_name or session_json.get('name', 'Imported Session')
            
            # Create new session ID
            session_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            session_file = os.path.join(self.sessions_dir, f"{session_id}.json")
            
            # Update session metadata
            session_json['name'] = name
            session_json['session_id'] = session_id
            session_json['imported_date'] = datetime.now().isoformat()
            
            # Save imported session
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_json, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Session imported: {name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to import session: {e}")
            return False
    
    def auto_save_session(self, session_state: Dict[str, Any]) -> bool:
        """Auto-save current session state"""
        try:
            auto_save_name = f"AutoSave_{datetime.now().strftime('%Y%m%d_%H%M')}"
            return self.save_session(auto_save_name, session_state)
        
        except Exception as e:
            logger.error(f"Auto-save failed: {e}")
            return False
    
    def get_session_info(self, session_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a session"""
        try:
            session_file = self._find_session_file(session_name)
            if not session_file:
                return None
            
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            state = session_data.get('state', {})
            
            # Calculate session statistics
            stats = {
                'documents': len(state.get('documents', [])),
                'flashcards': len(state.get('flashcards', [])),
                'summaries': len(state.get('summaries', [])),
                'chat_history': len(state.get('chat_history', [])),
                'has_study_plan': 'study_plan' in state
            }
            
            return {
                'name': session_data.get('name', session_name),
                'created_date': session_data.get('created_date', ''),
                'imported_date': session_data.get('imported_date'),
                'session_id': session_data.get('session_id', ''),
                'statistics': stats,
                'file_size': os.path.getsize(session_file)
            }
        
        except Exception as e:
            logger.error(f"Failed to get session info for '{session_name}': {e}")
            return None
    
    def backup_sessions(self) -> Optional[bytes]:
        """Create backup of all sessions"""
        try:
            # Create temporary directory for backup
            with tempfile.TemporaryDirectory() as temp_dir:
                backup_dir = os.path.join(temp_dir, 'studymate_backup')
                os.makedirs(backup_dir)
                
                # Copy all session files
                if os.path.exists(self.sessions_dir):
                    for filename in os.listdir(self.sessions_dir):
                        if filename.endswith('.json'):
                            src = os.path.join(self.sessions_dir, filename)
                            dst = os.path.join(backup_dir, filename)
                            shutil.copy2(src, dst)
                
                # Create backup metadata
                backup_info = {
                    'backup_date': datetime.now().isoformat(),
                    'version': '1.0',
                    'sessions_count': len(os.listdir(backup_dir)) if os.path.exists(backup_dir) else 0
                }
                
                with open(os.path.join(backup_dir, 'backup_info.json'), 'w') as f:
                    json.dump(backup_info, f, indent=2)
                
                # Create ZIP archive
                backup_file = os.path.join(temp_dir, 'studymate_backup.zip')
                shutil.make_archive(backup_file[:-4], 'zip', backup_dir)
                
                # Read and return backup data
                with open(backup_file, 'rb') as f:
                    return f.read()
        
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return None
    
    def restore_sessions(self, backup_data: bytes) -> bool:
        """Restore sessions from backup"""
        try:
            # Create temporary directory for extraction
            with tempfile.TemporaryDirectory() as temp_dir:
                backup_file = os.path.join(temp_dir, 'backup.zip')
                
                # Save backup data to file
                with open(backup_file, 'wb') as f:
                    f.write(backup_data)
                
                # Extract backup
                import zipfile
                with zipfile.ZipFile(backup_file, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Find extracted backup directory
                backup_dir = None
                for item in os.listdir(temp_dir):
                    item_path = os.path.join(temp_dir, item)
                    if os.path.isdir(item_path) and item != '__pycache__':
                        backup_dir = item_path
                        break
                
                if not backup_dir:
                    logger.error("No backup directory found in archive")
                    return False
                
                # Verify backup
                backup_info_file = os.path.join(backup_dir, 'backup_info.json')
                if os.path.exists(backup_info_file):
                    with open(backup_info_file, 'r') as f:
                        backup_info = json.load(f)
                    logger.info(f"Restoring backup from {backup_info.get('backup_date', 'unknown date')}")
                
                # Restore session files
                restored_count = 0
                for filename in os.listdir(backup_dir):
                    if filename.endswith('.json') and filename != 'backup_info.json':
                        src = os.path.join(backup_dir, filename)
                        dst = os.path.join(self.sessions_dir, filename)
                        
                        # Avoid overwriting existing sessions
                        if os.path.exists(dst):
                            base, ext = os.path.splitext(filename)
                            counter = 1
                            while os.path.exists(dst):
                                dst = os.path.join(self.sessions_dir, f"{base}_restored_{counter}{ext}")
                                counter += 1
                        
                        shutil.copy2(src, dst)
                        restored_count += 1
                
                logger.info(f"Restored {restored_count} sessions")
                return True
        
        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            return False
    
    def _find_session_file(self, session_name: str) -> Optional[str]:
        """Find session file by name"""
        try:
            if not os.path.exists(self.sessions_dir):
                return None
            
            for filename in os.listdir(self.sessions_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.sessions_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            session_data = json.load(f)
                        
                        if session_data.get('name') == session_name:
                            return filepath
                    
                    except Exception as e:
                        logger.warning(f"Failed to read session file {filename}: {e}")
            
            return None
        
        except Exception as e:
            logger.error(f"Error finding session file: {e}")
            return None
    
    def _serialize_session_state(self, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize session state for storage"""
        serialized = {}
        
        # List of keys to serialize
        keys_to_save = [
            'documents', 'flashcards', 'summaries', 'chat_history',
            'study_plan', 'extracted_tables', 'image_analysis_results',
            'customized_content', 'goal_completions', 'personal_goals',
            'search_results', 'bookmarks', 'progress_history'
        ]
        
        for key in keys_to_save:
            if key in session_state:
                try:
                    # Convert complex objects to JSON-serializable format
                    value = session_state[key]
                    
                    if key == 'study_plan' and value:
                        # Handle study plan serialization
                        serialized[key] = self._serialize_study_plan(value)
                    else:
                        # Default serialization
                        serialized[key] = self._deep_serialize(value)
                
                except Exception as e:
                    logger.warning(f"Failed to serialize {key}: {e}")
        
        return serialized
    
    def _deserialize_session_state(self, serialized_state: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize session state from storage"""
        deserialized = {}
        
        for key, value in serialized_state.items():
            try:
                if key == 'study_plan' and value:
                    # Handle study plan deserialization
                    deserialized[key] = self._deserialize_study_plan(value)
                else:
                    # Default deserialization
                    deserialized[key] = value
            
            except Exception as e:
                logger.warning(f"Failed to deserialize {key}: {e}")
        
        return deserialized
    
    def _serialize_study_plan(self, study_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize study plan with special handling for dates"""
        serialized = study_plan.copy()
        
        # Convert datetime objects to ISO strings
        if 'overview' in serialized:
            overview = serialized['overview']
            if 'created_date' in overview and hasattr(overview['created_date'], 'isoformat'):
                overview['created_date'] = overview['created_date'].isoformat()
        
        return serialized
    
    def _deserialize_study_plan(self, study_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize study plan with special handling for dates"""
        return study_plan  # Dates will remain as ISO strings
    
    def _deep_serialize(self, obj: Any) -> Any:
        """Deep serialize complex objects"""
        if isinstance(obj, dict):
            return {k: self._deep_serialize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_serialize(item) for item in obj]
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):  # Custom objects
            return self._deep_serialize(obj.__dict__)
        else:
            return obj
    
    def cleanup_old_sessions(self, days_to_keep: int = 30) -> int:
        """Clean up sessions older than specified days"""
        try:
            if not os.path.exists(self.sessions_dir):
                return 0
            
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            deleted_count = 0
            
            for filename in os.listdir(self.sessions_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.sessions_dir, filename)
                    
                    try:
                        # Check file modification time
                        mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                        
                        if mod_time < cutoff_date:
                            # Also check session creation date
                            with open(filepath, 'r', encoding='utf-8') as f:
                                session_data = json.load(f)
                            
                            created_date_str = session_data.get('created_date', '')
                            if created_date_str:
                                created_date = datetime.fromisoformat(created_date_str.replace('Z', '+00:00'))
                                if created_date < cutoff_date:
                                    os.remove(filepath)
                                    deleted_count += 1
                                    logger.info(f"Deleted old session: {filename}")
                    
                    except Exception as e:
                        logger.warning(f"Error processing session file {filename}: {e}")
            
            return deleted_count
        
        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {e}")
            return 0
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            stats = {
                'total_sessions': 0,
                'total_size_bytes': 0,
                'oldest_session': None,
                'newest_session': None,
                'session_sizes': []
            }
            
            if not os.path.exists(self.sessions_dir):
                return stats
            
            session_dates = []
            
            for filename in os.listdir(self.sessions_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.sessions_dir, filename)
                    
                    try:
                        file_size = os.path.getsize(filepath)
                        stats['total_size_bytes'] += file_size
                        stats['total_sessions'] += 1
                        
                        # Get session info
                        with open(filepath, 'r', encoding='utf-8') as f:
                            session_data = json.load(f)
                        
                        session_name = session_data.get('name', filename[:-5])
                        created_date = session_data.get('created_date', '')
                        
                        stats['session_sizes'].append({
                            'name': session_name,
                            'size_bytes': file_size,
                            'created_date': created_date
                        })
                        
                        if created_date:
                            session_dates.append((session_name, created_date))
                    
                    except Exception as e:
                        logger.warning(f"Error reading session file {filename}: {e}")
            
            # Find oldest and newest sessions
            if session_dates:
                session_dates.sort(key=lambda x: x[1])
                stats['oldest_session'] = session_dates[0][0]
                stats['newest_session'] = session_dates[-1][0]
            
            return stats
        
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {'total_sessions': 0, 'total_size_bytes': 0}
