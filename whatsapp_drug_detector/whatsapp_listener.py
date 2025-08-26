# whatsapp_listener.py
import os
from dotenv import load_dotenv
load_dotenv()

import sys
import logging
import json
from datetime import datetime
from pathlib import Path
import shutil

from flask import Flask, request, jsonify
from flask_cors import CORS
from twilio.twiml.messaging_response import MessagingResponse

# Import our modules
from database import insert_message, init_db
from predict_local import predict_drug
from location_extractor import extract_location
from risk_score import compute_risk

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
FLASK_ENV = os.environ.get("FLASK_ENV", "development")
PORT = int(os.environ.get("PORT", 5000))
PRODUCTION_MODE = FLASK_ENV == "production"

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.absolute()
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

def setup_logging():
    log_level = logging.INFO if PRODUCTION_MODE else logging.DEBUG
    log_file = LOG_DIR / "app.log"
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    
    handlers = [file_handler]
    
    if not PRODUCTION_MODE:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        handlers.append(console_handler)
    
    logging.basicConfig(level=log_level, handlers=handlers, force=True)
    return logging.getLogger(__name__)

logger = setup_logging()

# -------------------------------------------------------------------
# Flask app
# -------------------------------------------------------------------
app = Flask(__name__)
CORS(app)  # allow dashboards to call API
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024  # 16KB max

# Add health check endpoint
@app.route("/healthz", methods=["GET"])
def healthz():
    """Simple health check for uptime monitoring"""
    return jsonify({"ok": True, "status": "healthy"})

@app.route("/", methods=["GET"])
def home():
    """Home endpoint with system info"""
    return jsonify({
        "status": "active",
        "message": "WhatsApp Drug Detection System",
        "version": "4.0",
        "mode": "production" if PRODUCTION_MODE else "development",
        "timestamp": datetime.now().isoformat()
    })

@app.route("/health", methods=["GET"])
def health_check():
    """Extended health check with disk usage"""
    try:
        total, used, free = shutil.disk_usage(BASE_DIR)
        free_mb = free / (1024 * 1024)
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "disk_free_mb": round(free_mb, 2),
            "production_mode": PRODUCTION_MODE
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route("/whatsapp", methods=["POST"])
def whatsapp():
    """Main WhatsApp webhook endpoint"""
    try:
        raw_msg = request.values.get("Body", "").strip()
        sender = request.values.get("From", "unknown_user")

        logger.info(f"Message from {sender[:20]}: {raw_msg[:50]}...")

        resp = MessagingResponse()

        # Handle empty messages
        if not raw_msg or len(raw_msg) < 2:
            resp.message("✅ Message received")
            return str(resp)

        # Sanitize message
        msg = raw_msg.replace('\0', '').replace('\r', '').strip()[:500]

        # --- Prediction and extraction ---
        is_suspicious_bert, confidence, drug_kw, context_kw, origin = predict_drug(msg)
        locations = extract_location(msg)
        
        # Compute risk based on detected drug keywords
        risk_info = compute_risk(confidence, locations, drug_keywords=drug_kw)
        
        # Use risk_info to determine suspicious flag
        is_suspicious_flag = 1 if risk_info['risk_score'] > 0 else 0

        # Store in database
        insert_message(
            sender=sender,
            message=msg,
            is_suspicious=1 if is_suspicious_flag else 0,
            confidence=confidence,
            drug_keywords=json.dumps(drug_kw),
            context_keywords=json.dumps(context_kw),
            locations=json.dumps(locations),
            risk_score=risk_info['risk_score'],
            detection_source=origin
        )

        # Prepare response based on risk_score
        if is_suspicious_flag:
            reply = (
                f"*🚨 Suspicious Message Detected:*\n"
                f"📩 Message: {msg}\n"
                f"🧪 Drug Terms: {json.dumps(drug_kw)}\n"
                f"🧠 Context: {json.dumps(context_kw)}\n"
                f"📍 Locations: {json.dumps(locations)}\n"
                f"🔢 Model Confidence: {round(confidence,3)}\n"
                f"📊 Risk Score: {risk_info['risk_score']}%\n"
                f"🟡 Risk Level: {risk_info['risk_level']}\n"
                f"🔍 Detection Method: {origin}"
            )
        else:
            reply = (
                f"*✅ Message is clear. No suspicious content detected.*\n"
                f"📩 Message: {msg}"
            )

        resp.message(reply)
        logger.info(f"Response sent to {sender[:20]} - Suspicious: {is_suspicious_flag}")
        return str(resp)

    except Exception as e:
        logger.error(f"Error in /whatsapp endpoint: {e}")
        resp = MessagingResponse()
        resp.message("*🔧 System temporarily unavailable*")
        return str(resp)

# -------------------------------------------------------------------
# Error handlers
# -------------------------------------------------------------------
@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file size too large"""
    return jsonify({"error": "Request too large"}), 413

@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

# -------------------------------------------------------------------
# Initialize database and run app
# -------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Initializing database...")
    init_db()
    logger.info("WhatsApp Drug Detection System v4.0 starting...")
    logger.info(f"Running in {'PRODUCTION' if PRODUCTION_MODE else 'DEVELOPMENT'} mode")
    logger.info(f"Server will start on host=0.0.0.0, port={PORT}")
    
    app.run(
        host="0.0.0.0", 
        port=PORT, 
        debug=not PRODUCTION_MODE, 
        threaded=True
    )
# Initialize DB regardless of dev/prod
logger.info("Initializing database...")
init_db()