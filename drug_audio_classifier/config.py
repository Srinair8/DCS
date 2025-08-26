import os
from typing import List, Optional

# Your existing constants (keeping them exactly as they are)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "drug_classifier_model")
DEFAULT_WHISPER_MODEL = "base"
DEFAULT_MAX_LENGTH = 256
DEFAULT_THRESHOLD = 0.70

DRUG_KEYWORDS = [
    "stuff", "package", "goods", "deal", "pick up", "pickup", "stash", "green",
    "weed", "pot", "coke", "cocaine", "white", "powder", "score", "high",
    "gram", "g", "pill", "tabs", "md", "mdma", "lsd", "charas", "hash", "ganja",
    "dope", "joint", "puff", "trip", "syringe", "needle", "gear", "supply",
    "quality", "batch", "hook me up", "hookup", "overdose", "rave", "party"
]

HIGH_RISK_KEYWORDS = [
    "coke", "cocaine", "weed", "pot", "tabs", "mdma", "lsd", "charas", "hash", 
    "ganja", "dope", "overdose", "syringe", "needle", "gear"
]

# Additional production configurations (extending your setup)
class ProductionConfig:
    def __init__(self):
        # Model settings (using your defaults)
        self.MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
        self.WHISPER_MODEL = os.getenv("WHISPER_MODEL", DEFAULT_WHISPER_MODEL)
        self.MAX_LENGTH = int(os.getenv("MAX_LENGTH", DEFAULT_MAX_LENGTH))
        self.THRESHOLD = float(os.getenv("THRESHOLD", DEFAULT_THRESHOLD))
        
        # Production settings
        self.MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
        self.MAX_AUDIO_DURATION = int(os.getenv("MAX_AUDIO_DURATION", "300"))  # 5 minutes
        self.ALLOWED_EXTENSIONS = ["wav", "mp3", "m4a", "flac", "ogg"]
        
        # Security settings
        self.RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))
        self.RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))  # 1 hour
        self.ENABLE_LOGGING = os.getenv("ENABLE_LOGGING", "true").lower() == "true"

# Global config instance
config = ProductionConfig()