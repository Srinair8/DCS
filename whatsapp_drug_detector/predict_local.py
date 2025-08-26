# predict_local.py
import os
import re
import json
import torch
import logging
import threading
from typing import Tuple, List, Optional
from functools import lru_cache
from collections import Counter
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Thread-safe logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DrugDetectionModel:
    """Thread-safe, optimized drug detection model"""
    
    def __init__(self, model_path: str = "model/saved_model"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._model_lock = threading.RLock()
        self._load_model()
        self._compile_patterns()
    
    def _load_model(self):
        """Load BERT model with retry mechanism"""
        if not os.path.exists(self.model_path):
            logger.warning(f"Model path {self.model_path} not found. Using rule-based detection only.")
            return
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with self._model_lock:
                    self.model = DistilBertForSequenceClassification.from_pretrained(self.model_path)
                    self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_path)
                    self.model.eval()
                    # Disable gradients for inference
                    for param in self.model.parameters():
                        param.requires_grad = False
                logger.info("✅ DistilBERT model loaded successfully")
                return
            except Exception as e:
                logger.error(f"Model loading attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error("Failed to load model after all retries. Using rule-based only.")
    
    def _compile_patterns(self):
        """Pre-compile all regex patterns for performance"""
        self.DRUG_KEYWORDS = {
            'high_risk_drugs': ['brown sugar', 'smack', 'heroin', 'cocaine', 'crack', 'ice', 'meth', 'crystal'],
            'cannabis': ['maal', 'ganja', 'charas', 'hash', 'weed', 'pot', 'marijuana', 'stuff'],
            'steroids': ['gear', 'juice', 'roids', 'steroid', 'cycle', 'tren', 'deca', 'steroids'],
            'prescription_abuse': ['tramadol', 'alprazolam', 'xanax', 'codeine', 'oxy', 'opioid'],
            'general_slang': ['supply', 'goods', 'product', 'material']
        }
        
        self.CONTEXT_KEYWORDS = {
            'transaction': ['available', 'arrange', 'supplier', 'price', 'selling', 'deal', 'buy'],
            'secrecy': ['no one home', 'parents away', 'private', 'secret', 'quietly', 'alone'],
            'meeting': ['meet', 'delivery', 'pick up', 'collect', 'come over'],
            'urgency': ['urgent', 'asap', 'tonight', 'immediately', 'quick'],
            'consumption': ['enjoy', 'party', 'relax', 'chill', 'get high', 'trip']
        }
        
        self.INNOCENT_KEYWORDS = [
            'traffic', 'bus', 'restaurant', 'food', 'movie', 'work', 'office', 
            'meeting', 'study', 'exam', 'hospital', 'doctor', 'medicine', 
            'pharmacy', 'prescription', 'medical', 'birthday', 'celebration'
        ]
        
        # Compile patterns once
        all_drug_words = [word for words in self.DRUG_KEYWORDS.values() for word in words]
        all_context_words = [word for words in self.CONTEXT_KEYWORDS.values() for word in words]
        
        self.drug_patterns = [re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE) for word in all_drug_words]
        self.context_patterns = [re.compile(re.escape(phrase), re.IGNORECASE) for phrase in all_context_words]
        self.innocent_patterns = [re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE) for word in self.INNOCENT_KEYWORDS]
    
    @lru_cache(maxsize=1000)
    def normalize_message(self, message: str) -> str:
        """Cached message normalization"""
        if not message or not isinstance(message, str):
            return ""
        
        # Security: Remove null bytes and limit length
        message = message.replace('\0', '').replace('\r', '')[:1000]
        message = message.lower()
        message = re.sub(r'[^\w\s]', ' ', message)
        message = re.sub(r'\s+', ' ', message)
        return message.strip()
    
    def extract_keywords(self, message: str, patterns: List) -> List[str]:
        """Extract keywords using pre-compiled patterns"""
        normalized_msg = self.normalize_message(message)
        found_keywords = []
        
        for pattern in patterns:
            matches = pattern.findall(normalized_msg)
            found_keywords.extend(matches)
        
        return list(set(kw.lower() for kw in found_keywords))
    
    def predict(self, text: str) -> Tuple[int, float, List[str], List[str], str]:
        """Main prediction function with optimizations"""
        # Input validation
        if not text or len(text.strip()) < 3:
            return 0, 0.0, [], [], "Input too short"
        
        # Limit input size for performance
        text = text[:500]
        
        # Extract keywords
        drug_keywords = self.extract_keywords(text, self.drug_patterns)
        context_keywords = self.extract_keywords(text, self.context_patterns)
        innocent_keywords = self.extract_keywords(text, self.innocent_patterns)
        
        # Strong innocent pattern detection
        if innocent_keywords and not drug_keywords:
            return 0, 0.0, [], context_keywords, "Innocent pattern detected"
        
        # Calculate rule-based confidence
        total_words = max(len(text.split()), 1)
        drug_density = len(drug_keywords) / total_words
        context_density = len(context_keywords) / total_words
        innocent_density = len(innocent_keywords) / total_words
        
        rule_confidence = 0.0
        rule_pred = 0
        
        if drug_keywords and innocent_density <= drug_density:
            rule_confidence = min(1.0, drug_density + (context_density * 0.5))
            rule_pred = 1
        
        # BERT prediction
        bert_pred, bert_conf = 0, 0.0
        if self.model and self.tokenizer:
            try:
                with self._model_lock:
                    inputs = self.tokenizer(
                        text, 
                        return_tensors="pt", 
                        truncation=True, 
                        padding=True, 
                        max_length=128
                    )
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                        bert_pred = torch.argmax(probs, dim=1).item()
                        bert_conf = probs[0][bert_pred].item()
            except Exception as e:
                logger.error(f"BERT prediction failed: {e}")
        
        # Decision fusion with improved logic
        if drug_keywords:
            final_pred = 1
            # Use higher confidence between BERT and rules
            if self.model and bert_pred == 1:
                final_conf = max(bert_conf, rule_confidence)
                origin = "BERT + Keywords"
            else:
                final_conf = rule_confidence
                origin = "Keywords detected"
        else:
            final_pred = bert_pred if self.model else 0
            final_conf = bert_conf if self.model else 0.0
            origin = "BERT model" if self.model else "No detection"
        
        return final_pred, final_conf, drug_keywords, context_keywords, origin

# Global model instance
_model_instance = None
_model_lock = threading.Lock()

def get_model_instance():
    """Get singleton model instance (thread-safe)"""
    global _model_instance
    if _model_instance is None:
        with _model_lock:
            if _model_instance is None:
                _model_instance = DrugDetectionModel()
    return _model_instance

def predict_drug(text: str) -> Tuple[int, float, List[str], List[str], str]:
    """Public interface for drug prediction"""
    model = get_model_instance()
    return model.predict(text)