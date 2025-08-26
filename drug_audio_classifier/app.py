# app.py 
import os
import re
import tempfile
import whisper
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import requests 
import logging
import traceback
from datetime import datetime  # FIXED: Added missing import
from typing import Optional, Tuple, Dict, Any

# =========================
# API CONFIGURATION - ADD THIS NEAR THE TOP
# =========================
@st.cache_data(ttl=60)  # Cache for 1 minute
def get_api_base():
    """Get API base URL from secrets or environment"""
    try:
        # Try Streamlit secrets first (for deployed apps)
        api_base = st.secrets.get("API_BASE", "")
        if api_base:
            return api_base
    except:
        pass
    
    # Fallback to environment variable (for local development)
    return os.getenv("API_BASE", "http://localhost:5000")

API_BASE = get_api_base()

# API Health Check Function
@st.cache_data(ttl=30)  # Cache for 30 seconds
def check_api_health():
    """Check if Flask API is healthy"""
    try:
        response = requests.get(f"{API_BASE}/healthz", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"error": f"HTTP {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return False, {"error": str(e)}

# Function to send analysis results to Flask API (optional)
def send_analysis_to_api(analysis_data):
    """Send analysis results to Flask API for storage"""
    try:
        response = requests.post(f"{API_BASE}/api/analysis", 
                               json=analysis_data, 
                               timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send analysis to API: {e}")
        return False

# Import your existing modules
from predict import predict, load_model
from config import config, DRUG_KEYWORDS, HIGH_RISK_KEYWORDS
from utils import (
    logger, security_manager, file_manager, model_manager,
    setup_production_logging
)

# Load model once at app startup
load_model(config.MODEL_PATH)

# Additional context patterns for better detection
DRUG_CONTEXT_PATTERNS = [
    r'(?i)(picked?\s*(it|them)\s*up|got\s*the\s*(stuff|package|goods))',
    r'(?i)(meet\s*(at|near|behind)|behind\s*the\s*(metro|station))',
    r'(?i)(too\s*risky|cops?\s*(were|are)\s*there)',
    r'(?i)(same\s*source|better\s*this\s*time)',
    r'(?i)(payment|pay|crypto|money|cash)\s*(through|via|using)',
    r'(?i)(bringing|getting|delivery)',
    r'(?i)(saturday|party|rave)',
    r'(?i)(mumbai|supplier)',
    r'(?i)(straight\s*from|coming\s*from)'
]

# Global whisper model
whisper_model = None

@st.cache_resource
def load_whisper_model():
    """Load Whisper model with production error handling"""
    global whisper_model
    if whisper_model is None:
        try:
            whisper_model = whisper.load_model(config.WHISPER_MODEL)
            logger.info(f"Loaded Whisper model: {config.WHISPER_MODEL}")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            st.error("🔥 Failed to load speech recognition model. Please contact system administrator.")
            st.stop()
    return whisper_model

def transcribe_audio_production(model, audio_path: str) -> str:
    """Production-grade audio transcription with comprehensive error handling"""
    try:
        # Validate file exists and is readable
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            raise ValueError("Audio file is empty")
        
        # Log transcription attempt
        logger.info(f"Starting transcription for file: {os.path.basename(audio_path)} ({file_size/1024/1024:.1f}MB)")
        
        # Transcribe with production settings
        result = model.transcribe(
            audio_path,
            fp16=False,  # More stable for production
            verbose=False,
            condition_on_previous_text=True,
            temperature=0.0  # More deterministic
        )
        
        transcription = result["text"].strip()
        
        if not transcription:
            logger.warning("Empty transcription result")
            return ""
        
        logger.info(f"Transcription completed: {len(transcription)} characters, detected language: {result.get('language', 'unknown')}")
        
        # Log transcription quality metrics
        confidence = result.get('segments', [{}])[0].get('avg_logprob', 0) if result.get('segments') else 0
        logger.info(f"Transcription confidence: {confidence:.3f}")
        
        return transcription
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        logger.error(traceback.format_exc())
        raise e

def simulate_conversation(transcribed_text: str) -> str:
    """Simulate conversation from transcribed text"""
    if not transcribed_text:
        return ""
    sentences = re.split(r'(?<=[?.!])\s+', transcribed_text.strip())
    convo_lines = []
    speaker = "A"
    for sentence in sentences:
        if sentence:
            convo_lines.append(f"{speaker}: {sentence}")
            speaker = "B" if speaker == "A" else "A"
    return "\n".join(convo_lines)

def highlight_drug_lines_html(conversation_text: str, keywords: list) -> Tuple[str, Dict]:
    """ENHANCED version with better keyword detection while keeping original logic"""
    if not conversation_text:
        return "", {}
    
    lines = conversation_text.split("\n")
    line_hits = {}
    highlighted_lines = []

    total_keyword_matches = 0
    
    for line in lines:
        hits = []
        for kw in sorted(keywords, key=len, reverse=True):
            if ' ' in kw:  # Multi-word keywords
                if re.search(rf'(?i){re.escape(kw)}', line):
                    hits.append(kw)
                    total_keyword_matches += 1
            else:  # Single word keywords  
                if re.search(rf'(?i)\b{re.escape(kw)}\b', line):
                    hits.append(kw)
                    total_keyword_matches += 1
        
        if hits:
            highlighted_lines.append(f"<p style='color:#e57373'><b>[DRUG]</b> {line}</p>")
            line_hits[line] = hits
        else:
            highlighted_lines.append(f"<p>{line}</p>")
    
    logger.info(f"Keyword detection: {total_keyword_matches} matches across {len(line_hits)} lines")
    return "".join(highlighted_lines), line_hits

def compute_enhanced_drug_score(text: str, conversation_text: str, detected_keywords: Dict) -> Tuple[float, int, int]:
    """
    ENHANCED drug detection scoring that considers:
    - Presence of high-risk keywords  
    - Context patterns
    - Keyword density
    - PLUS: Better keyword detection in full text
    """
    
    try:
        # Count keywords from detected_keywords
        high_risk_count = 0
        total_keyword_count = 0
        
        for line_keywords in detected_keywords.values():
            total_keyword_count += len(line_keywords)
            for kw in line_keywords:
                if kw.lower() in [hr.lower() for hr in HIGH_RISK_KEYWORDS]:
                    high_risk_count += 1
        
        # Check full text for missed keywords
        text_lower = text.lower()
        additional_high_risk = sum(1 for kw in HIGH_RISK_KEYWORDS if kw.lower() in text_lower)
        additional_total = sum(1 for kw in DRUG_KEYWORDS if kw.lower() in text_lower)
        
        # Use the higher count
        high_risk_count = max(high_risk_count, additional_high_risk)
        total_keyword_count = max(total_keyword_count, additional_total)
        
        # Keyword density
        total_words = len(text.split())
        keyword_density = total_keyword_count / max(total_words, 1)
        
        # Context pattern scoring
        context_score = 0
        matched_patterns = 0
        
        for pattern in DRUG_CONTEXT_PATTERNS:
            if re.search(pattern, text):
                context_score += 0.15
                matched_patterns += 1
        
        # Enhanced scoring calculation
        enhanced_score = 0
        
        # High-risk keywords heavily weighted
        if high_risk_count > 0:
            enhanced_score += min(high_risk_count * 0.4, 0.8)
        
        # General keyword density
        enhanced_score += min(keyword_density * 2, 0.2)
        
        # Context patterns
        enhanced_score += min(context_score, 0.5)
        
        # Normalize to 0-1
        enhanced_score = min(enhanced_score, 1.0)
        
        # Production logging
        logger.info(f"Enhanced scoring - High-risk: {high_risk_count}, "
                   f"Total: {total_keyword_count}, "
                   f"Density: {keyword_density:.3f}, "
                   f"Context: {context_score:.3f}, "
                   f"Score: {enhanced_score:.3f}, "
                   f"Patterns: {matched_patterns}")
        
        return enhanced_score, high_risk_count, total_keyword_count
        
    except Exception as e:
        logger.error(f"Enhanced scoring error: {e}")
        return 0.0, 0, 0


def compute_multimodal_risk(pred_label: int, pred_prob: float, text: str, 
                          simulated_text: str, detected_keywords: Dict) -> Tuple[float, int]:
    """
    ENHANCED multimodal risk assessment with more aggressive keyword-based detection
    (keeping original structure but improving thresholds)
    """
    
    try:
        enhanced_score, high_risk_count, total_keyword_count = compute_enhanced_drug_score(
            text, simulated_text, detected_keywords
        )
        
        # Weighting logic
        if high_risk_count >= 1:
            model_weight, keyword_weight = 0.2, 0.8
            decision_reason = f"High-risk keywords detected (count={high_risk_count})"
        elif total_keyword_count >= 3:
            model_weight, keyword_weight = 0.3, 0.7
            decision_reason = f"Strong keyword evidence (count={total_keyword_count})"
        elif high_risk_count >= 1 or total_keyword_count >= 2:
            model_weight, keyword_weight = 0.4, 0.6
            decision_reason = f"Moderate keyword evidence"
        else:
            model_weight, keyword_weight = 0.7, 0.3
            decision_reason = f"Relying on ML model"
        
        # Score combination
        risk_score = (model_weight * pred_prob) + (keyword_weight * enhanced_score)
        
        # Decision logic with production thresholds
        if high_risk_count >= 1:
            adjusted_pred_label = 1
            final_reason = f"DRUG - High-risk keywords: {high_risk_count}"
        elif enhanced_score >= 0.4:
            adjusted_pred_label = 1
            final_reason = f"DRUG - Strong keyword evidence: {enhanced_score:.3f}"
        elif enhanced_score >= 0.3 and pred_prob >= 0.2:
            adjusted_pred_label = 1
            final_reason = f"DRUG - Combined evidence: enhanced={enhanced_score:.3f}, ml={pred_prob:.3f}"
        elif pred_prob >= config.THRESHOLD:  # Use configurable threshold
            adjusted_pred_label = 1
            final_reason = f"DRUG - High ML confidence: {pred_prob:.3f}"
        else:
            adjusted_pred_label = 0
            final_reason = f"NON_DRUG - Low confidence: enhanced={enhanced_score:.3f}, ml={pred_prob:.3f}"
        
        # Risk score adjustment
        if adjusted_pred_label == 1 and risk_score < 0.5:
            risk_score = max(risk_score, 0.6)
        
        # Production logging
        logger.info(f"Risk assessment - {final_reason}, final_risk={risk_score:.4f}")
        
        return min(max(risk_score, 0.0), 1.0), adjusted_pred_label
        
    except Exception as e:
        logger.error(f"Risk assessment error: {e}")
        return 0.5, 0  # Safe defaults

def main():
    """Production main application"""
    try:
        # Initialize production logging
        setup_production_logging()
        
        # Security check
        rate_limit_ok, rate_limit_msg = security_manager.check_rate_limit()
        if not rate_limit_ok:
            st.error(f"🚫 {rate_limit_msg}")
            logger.warning(f"Rate limit block: {security_manager.get_client_id()}")
            st.stop()
            
        # Page configuration
        st.set_page_config(
            page_title="🚨 Drug Audio Analyzer", layout="wide",
            initial_sidebar_state="collapsed"
        )

        st.title("🚨 Audio-Based Drug Conversation Detection System")
        st.markdown(
            "This AI powered system analyzes uploaded conversations to detect potential drug-related content, "
            "highlight risk keywords, and provide actionable insights to the Karnataka Police."
        )

        # ADD API STATUS CHECK IN SIDEBAR
        with st.sidebar:
            st.header("🔒 System Information")
            
            # API Health Check
            is_healthy, health_data = check_api_health()
            
            if is_healthy:
                st.success("✅ Flask API Online")
                if "timestamp" in health_data:
                    st.caption(f"API Status: {health_data.get('status', 'Unknown')}")
            else:
                st.error("❌ Flask API Offline")
                st.caption(f"Error: {health_data.get('error', 'Unknown')}")
                st.info(f"Expected API at: {API_BASE}")
            
            st.markdown("---")
                
        # Model availability check
        model_available, model_msg = model_manager.validate_model_availability()
        if not model_available:
            st.error(f"⚠️ System Error: {model_msg}")
            st.info("Please contact the system administrator to resolve this issue.")
            logger.error(f"Model validation failed: {model_msg}")
            st.stop()
            
        # Sidebar with system info
        with st.sidebar:
            st.header("🔒 System Information")
            st.success("✅ System Status: Operational")
            st.info(f"📁 Max file size: {config.MAX_FILE_SIZE_MB}MB")
            st.info(f"🎵 Formats: {', '.join(config.ALLOWED_EXTENSIONS)}")
            st.info(f"⏱️ Max duration: {config.MAX_AUDIO_DURATION//60} minutes")
            st.info(f"🤖 ML Threshold: {config.THRESHOLD}")
            
            # Clear cache option
            if st.button("🗑️ Clear System Cache"):
                st.cache_resource.clear()
                st.success("Cache cleared successfully!")
        
        # File upload
        uploaded_file = st.file_uploader(
            "🎙 Upload an audio file (WAV or MP3) for analysis", 
            type=config.ALLOWED_EXTENSIONS,
            help=f"Supported formats: {', '.join(config.ALLOWED_EXTENSIONS)}. Maximum size: {config.MAX_FILE_SIZE_MB}MB"
        )
        
        if not uploaded_file:
            return
        
        # File validation
        file_valid, file_msg = security_manager.validate_file(uploaded_file)
        if not file_valid:
            st.error(f"❌ {file_msg}")
            logger.warning(f"File validation failed: {file_msg} for {security_manager.get_client_id()}")
            return
        
        st.success(f"✅ {file_msg}")
        
        # Create secure temporary file
        audio_path = file_manager.create_secure_temp_file(uploaded_file)
        if not audio_path:
            st.error("Failed to process uploaded file. Please try again.")
            return
            
        try:
            # Display audio player
            st.audio(uploaded_file)
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
                
            status_text.text("🔹 Initializing speech recognition...")
            progress_bar.progress(10)
            
            model = load_whisper_model()
            progress_bar.progress(30)

            # Transcription
            status_text.text("🔹 Transcribing audio content...")
            progress_bar.progress(50)
            
            transcription = transcribe_audio_production(model, audio_path)
            progress_bar.progress(70)
            
            # NEW: Early transcription validation with enhanced feedback
            if not transcription:
                st.error("⚠️ No transcription produced. Please check the audio file.")
                try:
                    os.unlink(audio_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {audio_path}: {e}")
                return

            if len(transcription) < 10:
                st.warning("⚠️ Transcription is very short. Classification may be unreliable.")

            # NEW: Display raw transcription early for user feedback
            st.subheader("📝 Raw Transcription")
            st.write(transcription)

            # Analysis phase
            status_text.text("🔹 Performing threat analysis...")
            progress_bar.progress(85)
            
            # Get initial prediction
            pred_label, raw_prob = predict(transcription)
            logger.info(f"ML prediction: {'DRUG' if pred_label == 1 else 'NON_DRUG'} (confidence: {raw_prob:.4f})")

            # Conversation simulation
            simulated_text = simulate_conversation(transcription)

            # NEW: Display simulated conversation
            st.subheader("🎭 Simulated Conversation")
            st.text(simulated_text)
            
            # Keyword analysis
            highlighted_html, detected_keywords = highlight_drug_lines_html(simulated_text, DRUG_KEYWORDS)
            
            # Enhanced risk assessment
            risk_score, adjusted_prediction = compute_multimodal_risk(
                pred_label, raw_prob, transcription, simulated_text, detected_keywords
            )
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis completed successfully!")

            # NEW: Enhanced Analysis Section before main results
            st.subheader("🔍 Enhanced Analysis")
            enhanced_score, high_risk_count, total_keyword_count = compute_enhanced_drug_score(
                transcription, simulated_text, detected_keywords
            )
            st.write(f"**High-Risk Keywords Detected:** {high_risk_count}")
            st.write(f"**Total Drug Keywords Detected:** {total_keyword_count}")
            st.write(f"**Enhanced Drug Score:** {enhanced_score:.2f}/1.0")
            
            # Results presentation
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            # Main result display
            if adjusted_prediction == 1:
                st.markdown(
                    """
                    <div style='padding: 1.5rem; background: linear-gradient(90deg, #ffebee 0%, #ffcdd2 100%); 
                                border-left: 6px solid #d32f2f; border-radius: 8px; margin: 1rem 0;'>
                        <h2 style='color: #c62828; margin: 0; display: flex; align-items: center;'>
                            🚨 DRUG-RELATED CONTENT DETECTED
                        </h2>
                        <p style='margin: 0.5rem 0 0 0; color: #5d4037; font-size: 1.1rem;'>
                            <strong>High-confidence detection of drug-related conversation patterns</strong>
                        </p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

                # NEW: Confidence assessment for drug predictions
                if enhanced_score >= 0.6:
                    confidence_level = "HIGH"
                    confidence_color = "red"
                elif enhanced_score >= 0.3:
                    confidence_level = "MEDIUM"
                    confidence_color = "orange"
                else:
                    confidence_level = "LOW"
                    confidence_color = "yellow"
                
                st.markdown(f"**Confidence Level:** <span style='color: {confidence_color}; font-weight: bold;'>{confidence_level}</span>", 
                           unsafe_allow_html=True)
            else:
                st.markdown(
                    """
                    <div style='padding: 1.5rem; background: linear-gradient(90deg, #e8f5e8 0%, #c8e6c9 100%); 
                                border-left: 6px solid #388e3c; border-radius: 8px; margin: 1rem 0;'>
                        <h2 style='color: #2e7d32; margin: 0; display: flex; align-items: center;'>
                            ✅ NO DRUG CONTENT DETECTED
                        </h2>
                        <p style='margin: 0.5rem 0 0 0; color: #2d5016; font-size: 1.1rem;'>
                            <strong>Conversation appears to be non-drug related</strong>
                        </p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

            # Metrics dashboard
            enhanced_score, high_risk_count, total_keywords = compute_enhanced_drug_score(
                transcription, simulated_text, detected_keywords
            )
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "🤖 ML Model Analysis ",
                    f"{raw_prob:.1%}",
                    f"{'Drug' if pred_label == 1 else 'Non-Drug'}"
                )
            
            with col2:
                st.metric(
                    "🎯 Enhanced Score Analysis",
                    f"{enhanced_score:.1%}",
                    f"{high_risk_count} high-risk"
                )
            
            with col3:
                st.metric(
                    "⚠️ Risk Level", 
                    f"{risk_score:.1%}",
                    "🔴 CRITICAL" if risk_score >= 0.7 else 
                    "🟠 HIGH" if risk_score >= 0.5 else
                    "🟡 MEDIUM" if risk_score >= 0.3 else "🟢 LOW"
                )
            
            with col4:
                st.metric(
                    "🔍 Keywords Found",
                    f"{total_keywords}",
                    f"{len(detected_keywords)} flagged lines"
                )

            # NEW: Drug highlights section for any drug prediction
            if adjusted_prediction == 1:
                st.subheader("💡 Drug-Related Lines Highlighted")
                st.markdown(highlighted_html, unsafe_allow_html=True)

                if detected_keywords:
                    st.subheader("🔍 Detected Keywords per Line")
                    for line, kws in detected_keywords.items():
                        # Highlight high-risk keywords (ENHANCED)
                        high_risk_kws = [kw for kw in kws if kw.lower() in [hr.lower() for hr in HIGH_RISK_KEYWORDS]]
                        regular_kws = [kw for kw in kws if kw not in high_risk_kws]
                        
                        display_text = f"**Line:** `{line}`\n"
                        if high_risk_kws:
                            display_text += f"🚨 **High-Risk Keywords:** {', '.join(high_risk_kws)}\n"
                        if regular_kws:
                            display_text += f"⚠️ **Other Keywords:** {', '.join(regular_kws)}"
                        
                        st.markdown(display_text)

            # NEW: Final Risk Assessment section
            st.subheader("🚨 Final Risk Assessment")
            st.write(f"**Overall Risk Score:** {risk_score:.2f}/1.0")

            # Determine risk level
            if risk_score >= 0.7:
                risk_level = "🔴 **CRITICAL RISK**"
            elif risk_score >= 0.5:
                risk_level = "🟠 **HIGH RISK**"
            elif risk_score >= 0.3:
                risk_level = "🟡 **MEDIUM RISK**"
            else:
                risk_level = "🟢 **LOW RISK**"
                
            st.markdown(f"**Risk Level:** {risk_level}")

            # NEW: Show comparison between ML and enhanced prediction
            if pred_label != adjusted_prediction:
                st.info(f"🔄 **Prediction Adjusted**: ML model predicted {'DRUG' if pred_label == 1 else 'NON_DRUG'}, "
                        f"but enhanced analysis adjusted it to {'DRUG' if adjusted_prediction == 1 else 'NON_DRUG'}")
                                                                
            # System analysis summary
            st.markdown("---")
            st.subheader("📈 Analysis Summary")
            
            # Create summary dataframe
            summary_data = {
                "Analysis Component": [
                    "ML Model Prediction", 
                    "Enhanced Prediction", 
                    "Overall Risk Score",
                    "High-Risk Keywords",
                    "Total Keywords Detected",
                    "Flagged Conversation Lines",
                    "Processing Status"
                ],
                "Result": [
                    f"{'DRUG' if pred_label == 1 else 'NON_DRUG'} ({raw_prob:.1%} confidence)",
                    f"{'DRUG' if adjusted_prediction == 1 else 'NON_DRUG'}",
                    f"{risk_score:.1%} ({'CRITICAL' if risk_score >= 0.7 else 'HIGH' if risk_score >= 0.5 else 'MEDIUM' if risk_score >= 0.3 else 'LOW'})",
                    str(high_risk_count),
                    str(total_keywords),
                    str(len(detected_keywords)),
                    "✅ Complete"
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            # FIXED: Proper indentation for API integration
            if adjusted_prediction == 1:  # If drug content detected
                analysis_data = {
                    "type": "audio_analysis",
                    "filename": uploaded_file.name,
                    "prediction": "DRUG",
                    "confidence": risk_score,
                    "keywords_detected": total_keywords,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Download analysis report
            if adjusted_prediction == 1:
                st.markdown("**📥 Export Analysis Report**")
                
                # Create detailed report
                report_data = {
                    "timestamp": [pd.Timestamp.now()],
                    "filename": [uploaded_file.name],
                    "file_size_mb": [uploaded_file.size / (1024*1024)],
                    "ml_prediction": ["DRUG" if pred_label == 1 else "NON_DRUG"],
                    "ml_confidence": [raw_prob],
                    "enhanced_prediction": ["DRUG" if adjusted_prediction == 1 else "NON_DRUG"],
                    "risk_score": [risk_score],
                    "high_risk_keywords": [high_risk_count],
                    "total_keywords": [total_keywords],
                    "flagged_lines": [len(detected_keywords)],
                    "transcription_length": [len(transcription)]
                }
                
                report_df = pd.DataFrame(report_data)
                csv_data = report_df.to_csv(index=False).encode("utf-8")
                
                st.download_button(
                    label="📄 Download Analysis Report (CSV)",
                    data=csv_data,
                    file_name=f"drug_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

            # Debug section
            with st.expander("🐛 Debug Information (Click to expand)"):
                st.write("**Text being analyzed:**")
                st.code(transcription)
                
                detected_keywords_full = [kw for kw in DRUG_KEYWORDS if kw.lower() in transcription.lower()]
                detected_high_risk = [kw for kw in HIGH_RISK_KEYWORDS if kw.lower() in transcription.lower()]
                
                st.write(f"**All keywords found in full text:** {detected_keywords_full}")
                st.write(f"**High-risk keywords found:** {detected_high_risk}")
                st.write(f"**Line-by-line detection:** {detected_keywords}")
                
                # Check context patterns
                matched_contexts = []
                for pattern in DRUG_CONTEXT_PATTERNS:
                    if re.search(pattern, transcription):
                        matched_contexts.append(pattern)
                st.write(f"**Context patterns matched:** {len(matched_contexts)}")

        except Exception as e:
            logger.error(f"Processing error: {e}")
            logger.error(traceback.format_exc())
            st.error(f"❌ Processing failed: {str(e)}")
        
        finally:
            # Cleanup temporary file
            try:
                if audio_path and os.path.exists(audio_path):
                    os.unlink(audio_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {audio_path}: {e}")

    except Exception as e:
        logger.error(f"Application error: {e}")
        logger.error(traceback.format_exc())
        st.error("🔥 System error occurred. Please contact administrator.")  

if __name__ == "__main__":
    main()