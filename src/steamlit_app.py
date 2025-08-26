import streamlit as st
import urllib.parse
import requests  # ADD THIS IMPORT
import os

# =========================
# API CONFIGURATION - ADD THIS AT THE TOP
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

# Function to get system stats from API
@st.cache_data(ttl=60)  # Cache for 1 minute
def get_system_stats():
    """Get system statistics from Flask API"""
    try:
        response = requests.get(f"{API_BASE}/api/stats", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException as e:
        return None

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Drug Crime Intelligence System", layout="wide")

# --- HEADER ---
st.markdown("""
<div style="background: linear-gradient(90deg, #FF6B6B, #4ECDC4); 
            padding: 2rem; border-radius: 12px; color: white; text-align: center;">
    <h1>🚨 Drug Crime Intelligence System</h1>
    <p style="font-size:18px;">Select a module below to monitor drug activity in real-time</p>
</div>
""", unsafe_allow_html=True)

# ADD SYSTEM STATUS SECTION
st.markdown("<br>", unsafe_allow_html=True)

# System Status Bar
col1, col2, col3 = st.columns(3)

with col1:
    is_healthy, health_data = check_api_health()
    if is_healthy:
        st.success("✅ Flask API: Online")
    else:
        st.error("❌ Flask API: Offline")

with col2:
    # You can add WhatsApp dashboard health check here if needed
    st.info("📱 WhatsApp Module: Ready")

with col3:
    # You can add Audio dashboard health check here if needed
    st.info("🎤 Audio Module: Ready")

# Optional: Display system statistics
stats = get_system_stats()
if stats:
    st.markdown("### 📊 System Overview")
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("Total Messages", stats.get("total_messages", "N/A"))
    
    with metric_col2:
        st.metric("Suspicious Detected", stats.get("suspicious_count", "N/A"))
    
    with metric_col3:
        st.metric("Today's Activity", stats.get("today_count", "N/A"))
    
    with metric_col4:
        st.metric("Average Risk", f"{stats.get('avg_risk', 0):.1f}%")

st.markdown("<br>", unsafe_allow_html=True)

# --- MODULE CARDS ---
col1, col2 = st.columns(2, gap="large")

# WhatsApp Module
with col1:
    st.markdown("""
    <div style="background-color:#FF6B6B; padding:20px; border-radius:15px; text-align:center; height:250px;">
        <h2 style="color:white;">📱 WhatsApp Drug Detection</h2>
        <p style="color:white; font-size:16px;">Monitor and analyze WhatsApp messages for drug-related content.</p>
    </div>
    """, unsafe_allow_html=True)
    
    TWILIO_NUMBER = "+14155238886"
    whatsapp_url = f"https://web.whatsapp.com/send?phone={urllib.parse.quote(TWILIO_NUMBER)}"
    
    st.markdown(f"""
    <div style="text-align:center; margin-top:10px;">
        <a href="{whatsapp_url}" target="_blank">
            <button style="background-color:white; color:#FF6B6B; 
                           font-size:16px; padding:10px 20px; border-radius:10px; border:none; cursor:pointer;">
                Open WhatsApp Web
            </button>
        </a>
    </div>
    """, unsafe_allow_html=True)

    # Update these URLs to use deployed Streamlit app URLs when available
    dashboard_url = "http://localhost:8502"  # Replace with your deployed URL
    st.markdown(f"""
    <div style="text-align:center; margin-top:10px;">
        <a href="{dashboard_url}" target="_blank">
            <button style="background-color:white; color:#FF6B6B; 
                           font-size:16px; padding:10px 20px; border-radius:10px; border:none; cursor:pointer;">
                Open WhatsApp Dashboard
            </button>
        </a>
    </div>
    """, unsafe_allow_html=True)

# Audio Module
with col2:
    st.markdown("""
    <div style="background-color:#4ECDC4; padding:20px; border-radius:15px; text-align:center; height:250px;">
        <h2 style="color:white;">🎤 Audio Drug Detection</h2>
        <p style="color:white; font-size:16px;">Analyze audio messages in real-time to detect drug-related threats.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Update this URL to use deployed Streamlit app URL when available
    audio_url = "http://localhost:8503"  # Replace with your deployed URL
    st.markdown(f"""
    <div style="text-align:center; margin-top:50px;">
        <a href="{audio_url}" target="_blank">
            <button style="background-color:white; color:#4ECDC4; 
                           font-size:16px; padding:10px 20px; border-radius:10px; border:none; cursor:pointer;">
                Open Audio Dashboard
            </button>
        </a>
    </div>
    """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align:center; color:gray; font-size:14px;">
    Flask API Status: {'🟢 Online' if is_healthy else '🔴 Offline'} | 
    API Endpoint: <a href='{API_BASE}'>{API_BASE}</a>
</div>
""", unsafe_allow_html=True)