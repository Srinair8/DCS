#location_extractor.py
 
import spacy
from functools import lru_cache
from typing import List

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class LocationExtractor:
    """Optimized location extraction with caching"""
    
    def __init__(self):
        self.nlp = self._load_spacy_model()
    
    def _load_spacy_model(self):
        """Load spaCy model with error handling"""
        try:
            nlp = spacy.load("en_core_web_sm")
            # Disable unnecessary components for performance
            nlp.disable_pipes(["tagger", "parser", "attribute_ruler", "lemmatizer"])
            return nlp
        except OSError:
            logger.warning("spaCy model not found. Location extraction disabled.")
            return None
    
    @lru_cache(maxsize=500)
    def extract_location(self, text: str) -> List[str]:
        """Extract locations with caching for performance"""
        if not text or not text.strip() or not self.nlp:
            return []
        
        # Limit text length for performance
        text = text[:200]
        
        found_locations = set()
        
        try:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ("GPE", "LOC", "FAC"):  # Geographic entities
                    ent_text = ent.text.strip()
                    # Filter valid locations
                    if (len(ent_text) >= 2 and 
                        any(c.isalpha() for c in ent_text) and
                        not ent_text.lower() in ['home', 'here', 'there', 'place']):
                        found_locations.add(ent_text.title())
        except Exception as e:
            logger.error(f"Location extraction failed: {e}")
        
        return list(found_locations)

# Global extractor instance
_extractor_instance = None

def extract_location(text: str) -> List[str]:
    """Public interface for location extraction"""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = LocationExtractor()
    return _extractor_instance.extract_location(text)