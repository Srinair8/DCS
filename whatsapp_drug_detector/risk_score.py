#risk_score.py
from typing import Dict, List, Union, Optional

def compute_risk(
    confidence: float, 
    locations: Optional[List[str]] = None, 
    drug_keywords: Optional[List[str]] = None, 
    context_keywords: Optional[List[str]] = None
) -> Dict[str, Union[float, str, bool, List[str]]]:
    """
    Compute risk score with improved accuracy and performance
    """
    locations = locations or []
    drug_keywords = drug_keywords or []
    context_keywords = context_keywords or []
    
    # CRITICAL: No drugs = no risk
    if not drug_keywords:
        return {
            "risk_score": 0,
            "risk_level": "CLEAR",
            "confidence": round(confidence, 3),
            "locations": locations,
            "is_suspicious": False
        }
    
    # Base risk from model confidence
    base_risk = confidence * 100
    
    # Enhanced location risk assessment
    location_multiplier = 1.0
    high_risk_locations = {
        'railway station': 0.15, 'train station': 0.15,
        'park': 0.10, 'club': 0.20, 'bar': 0.15,
        'hostel': 0.10, 'college': 0.12, 'school': 0.12,
        'bus stop': 0.08, 'metro': 0.08
    }
    
    for location in locations:
        location_lower = location.lower()
        for risk_loc, multiplier in high_risk_locations.items():
            if risk_loc in location_lower:
                location_multiplier += multiplier
                break
    
    # Enhanced context risk assessment
    context_multiplier = 1.0
    high_risk_contexts = {
        'cash': 0.15, 'money': 0.10, 'payment': 0.08,
        'hide': 0.20, 'secret': 0.15, 'private': 0.10,
        'meet': 0.12, 'delivery': 0.18, 'pickup': 0.15,
        'package': 0.12, 'stuff': 0.10, 'supply': 0.15,
        'urgent': 0.08, 'asap': 0.08, 'tonight': 0.10
    }
    
    for context in context_keywords:
        context_lower = context.lower()
        for risk_ctx, multiplier in high_risk_contexts.items():
            if risk_ctx in context_lower:
                context_multiplier += multiplier
    
    # Drug type risk weighting
    drug_risk_multiplier = 1.0
    high_risk_drugs = ['heroin', 'cocaine', 'meth', 'crystal', 'brown sugar', 'smack']
    medium_risk_drugs = ['weed', 'hash', 'ganja', 'marijuana']
    
    for drug in drug_keywords:
        drug_lower = drug.lower()
        if any(hr_drug in drug_lower for hr_drug in high_risk_drugs):
            drug_risk_multiplier += 0.25
        elif any(mr_drug in drug_lower for mr_drug in medium_risk_drugs):
            drug_risk_multiplier += 0.10
    
    # Calculate final risk score
    risk_score = base_risk * location_multiplier * context_multiplier * drug_risk_multiplier
    risk_score = min(round(risk_score, 1), 100.0)  # Cap at 100
    
    # Determine risk level with better thresholds
    if risk_score >= 85:
        risk_level = "CRITICAL"
    elif risk_score >= 70:
        risk_level = "HIGH"
    elif risk_score >= 50:
        risk_level = "MEDIUM"
    elif risk_score >= 25:
        risk_level = "LOW"
    else:
        risk_level = "MINIMAL"
    
    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "confidence": round(confidence, 3),
        "locations": locations,
        "is_suspicious": True
    }