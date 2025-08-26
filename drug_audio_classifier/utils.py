import os
import logging
import tempfile
import hashlib
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import streamlit as st
from config import config
from pydub import AudioSegment
import json
import threading

# Setup logging
def setup_production_logging():
    """Setup production-grade logging"""
    if config.ENABLE_LOGGING:
        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        
        # Configure logging
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        
        # File handler with rotation
        file_handler = logging.FileHandler("logs/drug_detector.log")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            handlers=[file_handler, console_handler]
        )
        
        logger = logging.getLogger(__name__)
        logger.info("Production logging initialized")
        return logger
    else:
        return logging.getLogger(__name__)

logger = setup_production_logging()

RATE_LIMIT_FILE = "rate_limit.json"
rate_limit_lock = threading.Lock()

class SecurityManager:
    """Production security management"""
    def __init__(self):
        self.request_counts: Dict[str, list] = {}
    
    def get_client_id(self) -> str:
        """Get client identifier for rate limiting"""
        try:
            # Use session state for client identification
            if 'client_id' not in st.session_state:
                st.session_state.client_id = hashlib.md5(
                    str(time.time()).encode()
                ).hexdigest()[:16]
            return st.session_state.client_id
        except:
            return "anonymous"
    
    def check_rate_limit(self) -> Tuple[bool, Optional[str]]:
        """Check if request is within rate limits"""
        try:
            client_id = self.get_client_id()
            current_time = time.time()

            # Load existing counts
            with rate_limit_lock:
                if os.path.exists(RATE_LIMIT_FILE):
                    with open(RATE_LIMIT_FILE, "r") as f:
                        self.request_counts = json.load(f)
                else:
                    self.request_counts = {}
            
            # Initialize client requests if not exists
            if client_id not in self.request_counts:
                self.request_counts[client_id] = []
            
            # Remove old requests outside the window
            self.request_counts[client_id] = [
                req_time for req_time in self.request_counts[client_id]
                if current_time - req_time < config.RATE_LIMIT_WINDOW
            ]
            
            # Check rate limit
            if len(self.request_counts[client_id]) >= config.RATE_LIMIT_REQUESTS:
                logger.warning(f"Rate limit exceeded for client: {client_id}")
                return False, f"Rate limit exceeded. Maximum {config.RATE_LIMIT_REQUESTS} requests per hour."
            
            # Add current request
            self.request_counts[client_id].append(current_time)

            # Save updated counts
            with rate_limit_lock:
                with open(RATE_LIMIT_FILE, "w") as f:
                    json.dump(self.request_counts, f)

            return True, None
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return True, None  # Allow request on error
    
    def validate_file(self, uploaded_file) -> Tuple[bool, str]:
        """Comprehensive file validation"""
        try:
            # Check file size
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > config.MAX_FILE_SIZE_MB:
                logger.warning(f"File too large: {file_size_mb:.1f}MB from {self.get_client_id()}")
                return False, f"File too large: {file_size_mb:.1f}MB (max: {config.MAX_FILE_SIZE_MB}MB)"
            
            # Check file extension
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension not in config.ALLOWED_EXTENSIONS:
                logger.warning(f"Invalid file type: {file_extension} from {self.get_client_id()}")
                return False, f"Unsupported file type: {file_extension}"
            
            # Check filename for malicious patterns
            if any(char in uploaded_file.name for char in ['..', '/', '\\']):
                logger.warning(f"Suspicious filename: {uploaded_file.name} from {self.get_client_id()}")
                return False, "Invalid filename"

            # Check audio duration
            uploaded_file.seek(0)
            audio = AudioSegment.from_file(uploaded_file, format=file_extension)
            duration_sec = len(audio) / 1000
            if duration_sec > config.MAX_AUDIO_DURATION:
                logger.warning(f"Audio too long: {duration_sec:.1f}s from {self.get_client_id()}")
                return False, f"Audio too long: {duration_sec:.1f}s (max: {config.MAX_AUDIO_DURATION}s)"
            
            logger.info(f"File validated: {uploaded_file.name} ({file_size_mb:.1f}MB, {duration_sec:.1f}s)")
            return True, "File validated successfully"
            
        except Exception as e:
            logger.error(f"File validation error: {e}")
            return False, f"Validation failed: {str(e)}"

class FileManager:
    """Secure file handling for production"""
    
    @staticmethod
    def create_secure_temp_file(uploaded_file) -> Optional[str]:
        """Create temporary file with security measures"""
        try:
            # Create secure temp directory
            temp_dir = "temp_audio"
            os.makedirs(temp_dir, exist_ok=True, mode=0o700)  # Secure permissions
            
            # Generate secure filename
            file_hash = hashlib.md5(
                f"{uploaded_file.name}{time.time()}".encode()
            ).hexdigest()[:12]
            
            file_extension = uploaded_file.name.split('.')[-1].lower()
            secure_filename = f"audio_{file_hash}.{file_extension}"
            
            # Create temporary file
            temp_path = os.path.join(temp_dir, secure_filename)
            
            uploaded_file.seek(0)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            
            # Set secure permissions
            os.chmod(temp_path, 0o600)
            
            logger.info(f"Created secure temp file: {secure_filename}")
            return temp_path
            
        except Exception as e:
            logger.error(f"Failed to create temp file: {e}")
            return None
    
    @staticmethod
    def cleanup_temp_file(file_path: str):
        """Securely cleanup temporary file"""
        try:
            if file_path and os.path.exists(file_path):
                # Secure deletion (overwrite then delete)
                if os.path.getsize(file_path) > 0:
                    with open(file_path, "r+b") as f:
                        length = f.seek(0, 2)  # Get file size
                        f.seek(0)
                        f.write(b'\x00' * length)  # Overwrite with zeros
                        f.flush()
                        os.fsync(f.fileno())
                
                os.unlink(file_path)
                logger.info(f"Securely deleted temp file: {os.path.basename(file_path)}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {file_path}: {e}")

class ModelManager:
    """Production model management"""
    
    @staticmethod
    def validate_model_availability() -> Tuple[bool, str]:
        """Check if model is properly deployed"""
        try:
            model_path = config.MODEL_PATH
            
            if not os.path.exists(model_path):
                return False, f"Model directory not found: {model_path}"
            
            # Check required files
            required_files = [
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json"
            ]
            
            # Accept either pytorch_model.bin or model.safetensors
            if os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
                required_files.append("pytorch_model.bin")
            elif os.path.exists(os.path.join(model_path, "model.safetensors")):
                required_files.append("model.safetensors")
            else:
                return False, "Missing model weight file: pytorch_model.bin or model.safetensors"
            
            # Check all required files exist
            missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
            
            if missing_files:
                return False, f"Missing model files: {', '.join(missing_files)}"
            
            logger.info(f"Model validation successful: {model_path}")
            return True, "Model ready"
            
        except Exception as e:
            logger.error(f"Model validation error: {e}")
            return False, f"Model validation failed: {str(e)}"

# Global instances
security_manager = SecurityManager()
file_manager = FileManager()
model_manager = ModelManager()
