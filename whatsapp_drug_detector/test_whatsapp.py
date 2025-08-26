from predict_local import predict_drug
from location_extractor import extract_location
from risk_score import compute_risk
import json

test_messages = [
    "Hi, I have some ganja and steroids, want to enjoy tonight",
    "Going to the hospital for a checkup",
    "Meet me tonight, parents away, bring the gear",
    "Office meeting at 3pm in Koramangala",
    "Let's chill and get high at my place",
    "Can you arrange some smack for tomorrow?",
    "Dinner at the restaurant with friends",
    "Need to buy some juice and roids ASAP",
    "Study session at the library",
    "Pickup delivery of the stuff tonight",
    "Birthday celebration at home, no work tomorrow",
    "Check out the new weed I got"
]

for msg in test_messages:
    is_suspicious, confidence, drug_kw, context_kw, origin = predict_drug(msg)
    locations = extract_location(msg)
    risk_info = compute_risk(confidence, locations)
    
    if is_suspicious:
        print(
            f"📩 Message: {msg}\n"
            f"🧪 Drug Terms: {json.dumps(drug_kw)}\n"
            f"🧠 Context: {json.dumps(context_kw)}\n"
            f"📍 Locations: {json.dumps(locations)}\n"
            f"🔢 Model Confidence: {round(confidence,3)}\n"
            f"📊 Risk Score: {risk_info['risk_score']}%\n"
            f"🟡 Risk Level: {risk_info['risk_level']}\n"
            f"🔍 Detection Method: {origin}\n"
            + "-"*60
        )
    else:
        print(
            f"📩 Message: {msg}\n"
            f"✅ Message is clear. No suspicious content detected.\n"
            + "-"*60
        )
