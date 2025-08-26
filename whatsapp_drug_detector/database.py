# database.py
import sqlite3
from pathlib import Path
import pandas as pd
import logging
import json

logger = logging.getLogger(__name__)

# --- Database Path ---
BASE_DIR = Path(__file__).parent.absolute()
DB_PATH = BASE_DIR / "whatsapp_messages.db"

def init_db():
    """Initialize the SQLite database and create table if not exists."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS whatsapp_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender TEXT NOT NULL,
                message TEXT NOT NULL,
                is_suspicious INTEGER DEFAULT 0,
                confidence REAL DEFAULT 0.0,
                drug_keywords TEXT,
                context_keywords TEXT,
                locations TEXT,
                risk_score REAL DEFAULT 0.0,
                detection_source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            conn.commit()
            logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

def insert_message(sender, message, is_suspicious=0, confidence=0.0,
                   drug_keywords=None, context_keywords=None,
                   locations=None, risk_score=0.0, detection_source=None):
    """Insert a single WhatsApp message record into the database."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO whatsapp_messages
            (sender, message, is_suspicious, confidence, drug_keywords, 
             context_keywords, locations, risk_score, detection_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                sender,
                message,
                is_suspicious,
                confidence,
                json.dumps(drug_keywords) if drug_keywords else None,
                json.dumps(context_keywords) if context_keywords else None,
                json.dumps(locations) if locations else None,
                risk_score,
                detection_source
            ))
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to insert message: {e}")

def fetch_messages(limit=500):
    """
    Returns rows from the DB as tuples:
    (id, sender, message, is_suspicious, confidence, drug_keywords, context_keywords, locations, risk_score, detection_source)
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT id, sender, message, is_suspicious, confidence, drug_keywords, 
                   context_keywords, locations, risk_score, detection_source
            FROM whatsapp_messages
            ORDER BY created_at DESC
            LIMIT {limit}
        """)
        rows = []
        for r in cursor.fetchall():
            rows.append((
                r[0],                     # id
                r[1],                     # sender
                r[2],                     # message
                r[3],                     # is_suspicious
                r[4],                     # confidence
                json.loads(r[5]) if r[5] else [],  # drug_keywords
                json.loads(r[6]) if r[6] else [],  # context_keywords
                json.loads(r[7]) if r[7] else [],  # locations
                r[8],                     # risk_score
                r[9]                      # detection_source
            ))
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"Failed to fetch messages: {e}")
        return []
