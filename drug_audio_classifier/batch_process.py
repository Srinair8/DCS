import os
import re
import json
import csv
import argparse
from datetime import datetime
from typing import List, Tuple
import logging
from sklearn.metrics import classification_report
import pandas as pd

import whisper
from predict import predict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# Config
# ----------------------------
DEFAULT_AUDIO_DIR = "audio_sample"
DEFAULT_REPORT_DIR = "reports"
DEFAULT_THRESHOLD = 0.70

DRUG_KEYWORDS = [
    "stuff", "package", "goods", "deal", "pick up", "pickup", "stash", "green",
    "weed", "pot", "coke", "cocaine", "white", "powder", "score", "high",
    "gram", "g", "pill", "tabs", "md", "mdma", "lsd", "charas", "hash", "ganja",
    "dope", "joint", "puff", "trip", "syringe", "needle", "gear", "supply",
    "quality", "batch", "hook me up", "hookup", "overdose", "rave", "party"  # Added missing keywords
]

HIGH_RISK_KEYWORDS = [
    "coke", "cocaine", "weed", "pot", "tabs", "mdma", "lsd", "charas", "hash", 
    "ganja", "dope", "overdose", "syringe", "needle", "gear"
]
# ----------------------------
# Helpers
# ----------------------------
def load_whisper(model_size: str = "base"):
    print(f"🔊 Loading Whisper model '{model_size}' ...")
    logger.info(f"Loading Whisper model '{model_size}'")
    model = whisper.load_model(model_size)
    logger.info("Whisper model loaded successfully")
    return model

def transcribe_audio(model, audio_path: str) -> str:
    result = model.transcribe(audio_path)
    transcription = result.get("text", "").strip()
    logger.info(f"Transcription for {audio_path}: {transcription[:50]}... (length: {len(transcription)})")
    return transcription

def simulate_conversation(text: str) -> str:
    if not text:
        return ""
    sentences = re.split(r'(?<=[?.!])\s+', text.strip())
    speaker = "A"
    lines = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        lines.append(f"{speaker}: {s}")
        speaker = "B" if speaker == "A" else "A"
    return "\n".join(lines)

def highlight_keywords(text: str, keywords: List[str]) -> Tuple[str, List[str], dict]:
    if not text:
        return "", [], {}
    hits = set()
    lines = text.split("\n")
    line_hits = {}
    highlighted_lines = []

    for line in lines:
        line_specific_hits = []
        for kw in sorted(keywords, key=len, reverse=True):
            pattern = rf'(?i)\b{re.escape(kw)}\b'
            if re.search(pattern, line):
                line_specific_hits.append(kw)
                hits.add(kw)
        if line_specific_hits:
            line_hits[line] = line_specific_hits
            highlighted_line = line
            for kw in line_specific_hits:
                pattern = rf'(?i)\b{re.escape(kw)}\b'
                highlighted_line = re.sub(pattern, f"**[{kw}]**", highlighted_line)
            highlighted_lines.append(highlighted_line)
        else:
            highlighted_lines.append(line)

    highlighted_text = "\n".join(highlighted_lines)
    return highlighted_text, sorted(hits), line_hits

def compute_enhanced_drug_score(text, conversation_text, detected_keywords):
    """Enhanced drug detection scoring - same as app.py"""
    
    # Count different types of keywords
    high_risk_count = 0
    total_keyword_count = 0
    
    # Check for high-risk keywords in the full text
    for keyword in HIGH_RISK_KEYWORDS:
        if re.search(rf'(?i)\b{re.escape(keyword)}\b', text):
            high_risk_count += 1
    
    # Count total keywords detected
    for line_keywords in detected_keywords.values():
        total_keyword_count += len(line_keywords)
    
    # Calculate keyword density
    total_words = len(text.split())
    keyword_density = total_keyword_count / max(total_words, 1)
    
    # Context pattern scoring
    context_score = 0
    
    # Drug transaction patterns
    transaction_patterns = [
        r'(?i)(payment|pay|crypto|money|cash)\s+(through|via|using)',
        r'(?i)(bringing|getting|pick\s*up|delivery)',
        r'(?i)(saturday|party|rave|meet)',
        r'(?i)(mumbai|supplier|source)',
        r'(?i)(straight\s+from|coming\s+from)'
    ]
    
    for pattern in transaction_patterns:
        if re.search(pattern, text):
            context_score += 0.2
    
    # Calculate enhanced score
    enhanced_score = 0
    
    # High-risk keywords heavily weighted
    if high_risk_count > 0:
        enhanced_score += min(high_risk_count * 0.3, 0.7)
    
    # General keyword density
    enhanced_score += min(keyword_density * 2, 0.2)
    
    # Context patterns
    enhanced_score += min(context_score, 0.3)
    
    # Normalize to 0-1
    enhanced_score = min(enhanced_score, 1.0)
    
    return enhanced_score, high_risk_count, total_keyword_count

def compute_multimodal_risk(pred_label, pred_prob, text, simulated_text, detected_keywords):
    """Improved multimodal risk assessment - same as app.py"""
    
    # Get enhanced drug score
    enhanced_score, high_risk_count, total_keyword_count = compute_enhanced_drug_score(
        text, simulated_text, detected_keywords
    )
    
    # Adaptive weighting based on keyword evidence
    if high_risk_count >= 2 or total_keyword_count >= 4:
        model_weight = 0.3
        keyword_weight = 0.7
        logger.info("Strong keyword evidence detected - prioritizing keyword analysis")
    elif high_risk_count >= 1 or total_keyword_count >= 2:
        model_weight = 0.4
        keyword_weight = 0.6
        logger.info("Moderate keyword evidence detected")
    else:
        model_weight = 0.7
        keyword_weight = 0.3
        logger.info("Weak keyword evidence - relying more on ML model")
    
    # Combine scores
    risk_score = (model_weight * pred_prob) + (keyword_weight * enhanced_score)
    
    # Decision logic with enhanced thresholds
    if enhanced_score >= 0.6:
        adjusted_pred_label = 1
        logger.info(f"DRUG prediction due to strong keyword evidence (enhanced_score={enhanced_score:.3f})")
    elif enhanced_score >= 0.3 and pred_prob >= 0.2:
        adjusted_pred_label = 1
        logger.info(f"DRUG prediction due to combined evidence (enhanced_score={enhanced_score:.3f}, ml_prob={pred_prob:.3f})")
    elif pred_prob >= 0.6:
        adjusted_pred_label = 1
        logger.info(f"DRUG prediction due to high ML confidence (ml_prob={pred_prob:.3f})")
    else:
        adjusted_pred_label = 0
        logger.info(f"NON_DRUG prediction (enhanced_score={enhanced_score:.3f}, ml_prob={pred_prob:.3f})")
    
    # Ensure risk score reflects the prediction
    if adjusted_pred_label == 1 and risk_score < 0.5:
        risk_score = max(risk_score, 0.6)
    
    return min(max(risk_score, 0.0), 1.0), adjusted_pred_label

def safe_mkdir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def write_text_report(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"File: {payload['file']}\n")
        f.write(f"Processed At: {payload['processed_at']}\n")
        f.write(f"Label: {'DRUG' if payload['label'] == 1 else 'NON_DRUG'} (DRUG prob={payload['probability']:.4f}, threshold={payload['threshold']:.2f})\n")
        f.write(f"Risk Score: {payload['risk_score']:.2f}\n")
        f.write(f"Confidence Flag: {payload['confidence_flag']}\n")
        f.write(f"Keywords Detected ({len(payload['keywords'])}): {', '.join(payload['keywords']) or 'None'}\n")
        f.write(f"Keyword Hits per Line:\n")
        for line, kws in payload['keyword_lines'].items():
            f.write(f"  - {line}: {', '.join(kws)}\n")
        f.write("\n--- RAW TRANSCRIPTION ---\n")
        f.write(payload["transcription"] + "\n")
        f.write("\n--- HIGHLIGHTED TRANSCRIPTION ---\n")
        f.write(payload["highlighted_transcription"] + "\n")
        f.write("\n--- SIMULATED CONVERSATION (A/B) ---\n")
        f.write(payload["simulated_conversation"] + "\n")
        f.write("\n--- CLASSIFICATION REPORT ---\n")
        f.write(payload["classification_report"] + "\n")

def write_json(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def append_csv_summary(csv_path: str, row: dict, fieldnames: List[str]):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def process_file(model, audio_path: str, report_dir: str, threshold: float, ground_truth: str = None) -> dict:
    print(f"🎧 Processing: {audio_path}")
    transcription = transcribe_audio(model, audio_path)
    if not transcription:
        logger.warning(f"Skipping {audio_path}: Empty transcription")
        return {
            "file": os.path.basename(audio_path),
            "processed_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "label": "ERROR",
            "probability": 0.0,
            "risk_score": 0.0,
            "threshold": float(threshold),
            "confidence_flag": "ERROR: Empty transcription",
            "keywords": [],
            "keyword_lines": {},
            "transcription": "",
            "highlighted_transcription": "",
            "simulated_conversation": "",
            "classification_report": "",
            "report_txt": "",
            "report_json": "",
        }

    simulated = simulate_conversation(transcription)
    label, prob = predict(transcription)
    logger.info(f"Raw prediction for {audio_path}: label={'DRUG' if label == 1 else 'NON_DRUG'}, DRUG prob={prob:.4f}")

    if prob > 0.5 and label == 0:
        logger.error(f"Prediction mismatch: DRUG prob={prob:.4f} > 0.5 but label=NON_DRUG. Overriding to DRUG.")
        label = 1
    elif prob < 0.5 and label == 1:
        logger.error(f"Prediction mismatch: DRUG prob={prob:.4f} < 0.5 but label=DRUG. Overriding to NON_DRUG.")
        label = 0

    highlighted, hits, line_hits = highlight_keywords(simulated, DRUG_KEYWORDS)
    logger.info(f"Before risk adjustment: label={'DRUG' if label == 1 else 'NON_DRUG'}, DRUG prob={prob:.4f}")
    risk_score, adjusted_label = compute_multimodal_risk(label, prob, transcription, simulated, line_hits)
    
    enhanced_score, high_risk_count, total_keyword_count = compute_enhanced_drug_score(transcription, simulated, line_hits)
    logger.info(f"After risk adjustment: label={'DRUG' if adjusted_label == 1 else 'NON_DRUG'}, risk_score={risk_score:.4f}")
    
    confidence = max(prob, 1 - prob)
    conf_flag = "OK" if confidence >= threshold else "UNCERTAIN"

    y_pred = [adjusted_label]
    if ground_truth and ground_truth in ["DRUG", "NON_DRUG"]:
        y_true = [1 if ground_truth == "DRUG" else 0]
    else:
        y_true = [adjusted_label]
        logger.warning(f"No ground truth provided for {audio_path}. Using predicted label for report.")

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["NON_DRUG", "DRUG"],
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    classification_report_str = report_df.to_string()

    base = os.path.basename(audio_path)
    stamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {
        "file": base,
        "processed_at": stamp,
        "label": adjusted_label,
        "probability": float(prob),
        "risk_score": float(risk_score),
        "enhanced_score": float(enhanced_score), 
        "high_risk_keywords": high_risk_count,    
        "total_keywords": total_keyword_count,  
        "threshold": float(threshold),
        "confidence_flag": conf_flag,
        "keywords": hits,
        "keyword_lines": line_hits,
        "transcription": transcription,
        "highlighted_transcription": highlighted,
        "simulated_conversation": simulated,
        "classification_report": classification_report_str,
    }

    name_no_ext, _ = os.path.splitext(base)
    txt_path = os.path.join(report_dir, f"{name_no_ext}.txt")
    json_path = os.path.join(report_dir, f"{name_no_ext}.json")
    write_text_report(txt_path, payload)
    write_json(json_path, payload)

    payload["report_txt"] = txt_path
    payload["report_json"] = json_path
    return payload

def main():
    parser = argparse.ArgumentParser(description="Batch transcribe + classify audio files")
    parser.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR, help="Folder containing .wav/.mp3")
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR, help="Where to store reports")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Confidence threshold (0..1)")
    parser.add_argument("--model-size", default="base", choices=["tiny", "base", "small", "medium", "large"], help="Whisper model size")
    parser.add_argument("--ground-truth-csv", default=None, help="CSV with file names and ground truth labels (file, label)")
    args = parser.parse_args()

    audio_dir = args.audio_dir
    report_dir = args.report_dir
    threshold = args.threshold
    ground_truth_csv = args.ground_truth_csv

    if not os.path.isdir(audio_dir):
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    ground_truth = {}
    if ground_truth_csv and os.path.exists(ground_truth_csv):
        gt_df = pd.read_csv(ground_truth_csv)
        if 'file' in gt_df.columns and 'label' in gt_df.columns:
            ground_truth = dict(zip(gt_df['file'], gt_df['label']))
            logger.info(f"Loaded ground truth labels for {len(ground_truth)} files")
        else:
            logger.warning("Ground truth CSV must have 'file' and 'label' columns. Ignoring.")

    safe_mkdir(report_dir)
    wmodel = load_whisper(args.model_size)

    exts = (".wav", ".mp3", ".m4a", ".flac", ".ogg")
    files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.lower().endswith(exts)]
    files.sort()

    if not files:
        print(f"⚠️ No audio files found in: {audio_dir}")
        return

    csv_path = os.path.join(report_dir, "summary.csv")
    fields = [
        "file", "processed_at", "label", "probability", "risk_score", 
        "enhanced_score", "high_risk_keywords", "total_keywords",  # ADD THESE
        "threshold", "confidence_flag", "keywords", "report_txt", "report_json"
    ]

    for path in files:
        try:
            file_name = os.path.basename(path)
            gt_label = ground_truth.get(file_name, None)
            payload = process_file(wmodel, path, report_dir, threshold, gt_label)
            row = {
                "file": payload["file"],
                "processed_at": payload["processed_at"],
                "label": "DRUG" if payload["label"] == 1 else "NON_DRUG",
                "probability": f"{payload['probability']:.4f}",
                "risk_score": f"{payload['risk_score']:.2f}",
                "enhanced_score": f"{payload.get('enhanced_score', 0):.2f}",      
                "high_risk_keywords": payload.get("high_risk_keywords", 0),     
                "total_keywords": payload.get("total_keywords", 0),
                "threshold": f"{payload['threshold']:.2f}",
                "confidence_flag": payload["confidence_flag"],
                "keywords": ";".join(payload["keywords"]) if payload["keywords"] else "",
                "report_txt": payload["report_txt"],
                "report_json": payload["report_json"],
            }
            append_csv_summary(csv_path, row, fields)
        except Exception as e:
            err_row = {
                "file": os.path.basename(path),
                "processed_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "label": "ERROR",
                "probability": "",
                "risk_score": "",
                "enhanced_score": "",           
                "high_risk_keywords": "",       
                "threshold": f"{threshold:.2f}",
                "confidence_flag": f"ERROR: {type(e).__name__}",
                "keywords": "",
                "report_txt": "",
                "report_json": "",
            }
            append_csv_summary(csv_path, err_row, fields)
            print(f"❌ Error on {path}: {e}")

    print(f"\n✅ Done. Summary saved to: {csv_path}")
    print(f"📂 Per-file reports saved under: {report_dir}")

if __name__ == "__main__":
    main()