import streamlit as st
import folium
import os
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import pandas as pd
import spacy
from collections import defaultdict, Counter
import re
import requests
import logging
from datetime import datetime
import json
import time
from database import fetch_messages  # Ensure this returns all relevant DB columns
import matplotlib.pyplot as plt

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

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Streamlit Config ---
st.set_page_config(
    page_title="WhatsApp Drug Risk Dashboard",
    page_icon="🚔",
    layout="wide"
)

# --- Load spaCy model ---
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("spaCy model 'en_core_web_sm' not found.\nRun: python -m spacy download en_core_web_sm")
        st.stop()

nlp = load_spacy_model()

# --- Drug categories & location data ---
@st.cache_data
def get_drug_categories():
    return {
        "cannabis":["maal","stuff","charas","weed","hash","dope","joint","ganja","pot","mary jane","bhang","grass","green","bud","mal"],
        "heroin":["brown sugar","smack","chitta","safed powder","H","white horse","gard","horse","junk"],
        "opium":["afeem","doda","doda post","kali nagini","posta","poppy"],
        "cocaine":["coke","snow","white","charlie","crack","blow","flake","powder","rock"],
        "synthetic":["E","LSD","tabs","trip","acid","mdma","molly","ecstasy","ice","crystal","meth","glass","tik"],
        "prescription":["corex","spasmo proxyvon","alprax","alprazolam","nitrosun","n10","lomotil","tramadol","diazepam","lyrica","pregabalin"],
        "inhalants":["erazex","whitener","glue","correction fluid","thinner","solvent"],
        "steroids":["roids","gear","juice","sustanon","deca","testosterone"]
    }

@st.cache_data
def get_location_data():
    # Base city coordinates for mapping known locations
    return {
        "BTM":[12.9154,77.6101],"Majestic":[12.9762,77.5714],"Shivajinagar":[12.9844,77.6030],
        "MG Road":[12.9751,77.6063],"Indiranagar":[12.9719,77.6412],"Koramangala":[12.9352,77.6245],
        "Yelahanka":[13.1007,77.5963],"Whitefield":[12.9698,77.7499],"Rajajinagar":[12.9914,77.5544],
        "Banashankari":[12.9180,77.5730],"Hebbal":[13.0352,77.5910],"Jayanagar":[12.9254,77.5931],
        "Basavanagudi":[12.9430,77.5730],"Electronic City":[12.8450,77.6600],"HSR Layout":[12.9110,77.6414],
        "KR Market":[12.9627,77.5801],"Kengeri":[12.9186,77.4786],"Ramamurthy Nagar":[13.0247,77.6784],
        "RT Nagar":[13.0196,77.5890],"Marathahalli":[12.9569,77.7011]
    }

drug_categories = get_drug_categories()
loc_coords = get_location_data()

# --- Helper Functions ---
def highlight_drug_keywords(text):
    if not text: return text
    all_drugs = [drug for cat in drug_categories.values() for drug in cat]
    for drug in all_drugs:
        pattern = rf'\b{re.escape(drug)}\b'
        text = re.sub(pattern, f'**{drug.upper()}**', text, flags=re.IGNORECASE)
    return text

def get_risk_level_color(score):
    if score >= 80:
        return "🔴 CRITICAL"
    elif score >= 60:
        return "🟠 HIGH"
    elif score >= 40:
        return "🟡 MEDIUM"
    else:
        return "🟢 LOW"

def export_results_json(df, location_summary):
    return json.dumps({
        "timestamp": datetime.now().isoformat(),
        "total_messages": len(df),
        "unique_locations": df['Location'].nunique(),
        "results": df.to_dict('records'),
        "location_summary": {loc: {"avg_risk": round(sum(r)/len(r),2),
                                   "message_count": len(r),
                                   "total_risk": sum(r)} for loc,r in location_summary.items()}
    }, indent=2)

def extract_context(text):
    context_keywords = ["enjoy","party","weekend","free","alone","home"]
    return [word for word in context_keywords if re.search(rf'\b{word}\b', text, re.IGNORECASE)]

# --- UI Header ---
st.markdown("""
<div style="background: linear-gradient(90deg, #FF6B6B, #4ECDC4); padding:1rem; border-radius:10px; color:white; text-align:center;">
<h1>🚔 WhatsApp Chat Intelligence: Real-Time Drug Threat Detection</h1>
<p>Real-Time AI-Powered Monitoring of WhatsApp Messages for Drug Threat Detection</p>
</div>
""", unsafe_allow_html=True)

# Add API status in sidebar - INSERT THIS AFTER YOUR HEADER
with st.sidebar:
    st.header("🔗 System Status")
    
    # API Health Check
    is_healthy, health_data = check_api_health()
    
    if is_healthy:
        st.success("✅ Flask API Online")
        if "timestamp" in health_data:
            st.caption(f"Last check: {health_data.get('timestamp', 'Unknown')}")
    else:
        st.error("❌ Flask API Offline")
        st.caption(f"Error: {health_data.get('error', 'Unknown')}")
        st.info(f"Expected API at: {API_BASE}")
    
    st.markdown("---")

# --- Fetch messages ---
num_messages = st.number_input("Number of recent messages to fetch:", min_value=50, max_value=2000, value=50, step=50)
rows = fetch_messages(num_messages)


# Convert rows to DataFrame
df_rows = pd.DataFrame(rows, columns=[
    "id","sender","message_text","time","confidence",
    "drug_keywords","context_keywords","locations",
    "risk_score","detection_source"
])

# Drop exact duplicates by message text (you can also include sender/time if needed)
df_rows = df_rows.drop_duplicates(subset=["message_text"])

# Convert back to list of rows
rows = df_rows.values.tolist()

# --- Process messages ---
results = []
heatmap_data = []
location_summary = defaultdict(list)
drug_counter = Counter()
risk_counts = defaultdict(int)
time_series_data = defaultdict(int)

for row in rows:
    try:
        message_text = row[2] if len(row) > 2 else ""
        highlighted_msg = highlight_drug_keywords(message_text)
        risk_score = row[8] if len(row) > 8 and row[8] is not None else 0
        raw_locs = row[7] if len(row) > 7 and row[7] else []

        # Normalize locations
        if isinstance(raw_locs, str):
            locations = [loc.strip().title() for loc in raw_locs.split(",") if loc.strip()]
        elif isinstance(raw_locs, list):
            locations = [loc.strip().title() for loc in raw_locs if loc.strip()]
        else:
            locations = []

        detected_drugs = [
            drug for cat in drug_categories.values() 
            for drug in cat 
            if re.search(rf'\b{re.escape(drug)}\b', message_text, re.IGNORECASE)
        ]
        for d in detected_drugs:
            drug_counter[d.lower()] += 1

        risk_level_only = get_risk_level_color(risk_score).split()[1]
        risk_counts[risk_level_only] += 1

        if len(row) > 10 and row[10]:
            ts = pd.to_datetime(row[10])
            hour_str = ts.strftime("%Y-%m-%d %H:00")
            time_series_data[hour_str] += 1

        results.append({
            "Message": highlighted_msg,
            "Drug Terms": detected_drugs,
            "Context": extract_context(message_text),
            "Location": locations[0] if locations else "📍 Not Detected",
            "Confidence": f"{row[4]*100:.1f}%" if len(row) > 4 and row[4] is not None else "0%",
            "Risk Score": risk_score,
            "Risk Level": get_risk_level_color(risk_score),
            "RiskLevelOnly": risk_level_only,
            "Detection Method": row[9] if len(row) > 9 else "BERT Model"
        })

        for loc in locations:
            loc = loc.strip().title()
            location_summary[loc].append(risk_score)
            # Assign coordinates: use loc_coords if known, else fallback to city center
            lat, lon = loc_coords.get(loc, [12.9716, 77.5946])
    
            heatmap_data.append([lat, lon, risk_score])

    except Exception as e:
        logger.error(f"Error processing row: {e}")

df_results = pd.DataFrame(results)
if not df_results.empty:
    if 'RiskLevelOnly' not in df_results.columns:
        df_results['RiskLevelOnly'] = df_results['Risk Level'].apply(lambda x: x.split()[1])

# --- Sidebar Filters ---
all_locations = list(set(list(loc_coords.keys()) + list(location_summary.keys()) + ["📍 Not Detected"]))
location_filter = st.sidebar.multiselect(
    "Select Locations",
    options=all_locations,
    default=all_locations
)
risk_filter = st.sidebar.multiselect(
    "Select Risk Level",
    options=["LOW","MEDIUM","HIGH","CRITICAL"],
    default=["LOW","MEDIUM","HIGH","CRITICAL"]
)

# --- Apply filters ---
if not df_results.empty:
    # Get all unique locations from data for filtering
    available_locations = df_results['Location'].unique().tolist()
    
    # Filter based on selections
    df_filtered = df_results[
        df_results['RiskLevelOnly'].isin(risk_filter) &
        df_results['Location'].isin(location_filter)
    ]
else:
    df_filtered = df_results.copy()
    
# --- Clean Location column ---
def clean_location(loc):
    if not loc or loc in ["{}", "[]", "", "📍 Not Detected"]:
        return "📍 Not Detected"
    # Remove unwanted characters
    loc = str(loc).replace("{","").replace("}","").replace("[","").replace("]","").replace('"',"").strip()
    return loc if loc else "📍 Not Detected"

df_filtered['Location'] = df_filtered['Location'].apply(clean_location)

# --- Display Results Table with scrollable HTML ---
st.subheader("📋 WhatsApp Messages Analysis")
if not df_filtered.empty:
    # Convert lists to strings for display
    df_display = df_filtered.copy()
    df_display["Drug Terms"] = df_display["Drug Terms"].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    df_display["Context"] = df_display["Context"].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    
    # Create scrollable HTML table
    df_html = df_display[["Message","Drug Terms","Context","Location","Confidence","Risk Score","Risk Level","Detection Method"]].to_html(escape=False)
    st.markdown(
        f'<div style="overflow-x:auto; max-width:100%">{df_html}</div>',
        unsafe_allow_html=True
    )
else:
    st.write("No messages to display.")

col1, col2 = st.columns(2)
with col1:
    st.download_button(
        "📊 Download CSV",
        df_filtered.to_csv(index=False),
        file_name=f"drug_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
with col2:
    st.download_button(
        "📄 Download JSON",
        export_results_json(df_filtered, location_summary),
        file_name=f"drug_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

# --- Heatmap ---
if heatmap_data:
    st.subheader("🗺️ Risk Zone Heatmap")
    m = folium.Map(location=[12.9716, 77.5946], zoom_start=11, tiles='OpenStreetMap')
    
    HeatMap(heatmap_data, radius=15, blur=10, max_zoom=14,
            gradient={0.2:'blue',0.4:'lime',0.6:'orange',0.8:'red',1.0:'darkred'}).add_to(m)
    
    marker_cluster = MarkerCluster().add_to(m)
    for loc, risks in location_summary.items():
        loc_clean = loc.strip("[]").strip()
        lat, lon = loc_coords.get(loc_clean, [12.9716, 77.5946])
        for risk_score in risks:
            risk_level = get_risk_level_color(risk_score).split()[1]
            color = "red" if risk_score>=80 else "orange" if risk_score>=60 else "yellow" if risk_score>=40 else "green"
            
            text = f"{loc_clean}  Risk Score: {risk_score:.1f}  Risk Level: {risk_level}"
            
            folium.Marker(
                location=[lat, lon],
                popup=text,
                tooltip=text,
                icon=folium.Icon(color=color)
            ).add_to(marker_cluster)
    
    st_folium(m, width=700, height=400)

# --- High-Risk Locations ---
if location_summary:
    st.subheader("🎯 Top 10 High-Risk Locations")

    # Clean and filter locations
    filtered_locations = {}
    for loc, risks in location_summary.items():
        # Convert list-like strings to plain text
        if isinstance(loc, list) and loc:
            loc_clean = loc[0].strip()
        else:
            loc_clean = str(loc).strip()
            # Remove brackets and quotes
            loc_clean = loc_clean.replace("[","").replace("]","").replace('"','').strip()

        # Only keep valid locations
        if loc_clean and loc_clean.lower() not in ["not detected", ""]:
            filtered_locations[loc_clean] = risks

    if filtered_locations:
        # Sort by average risk descending and take top 10
        top_zones = sorted(
            filtered_locations.items(),
            key=lambda x: sum(x[1]) / len(x[1]),
            reverse=True
        )[:10]

        # Prepare table data
        table_data = []
        for i, (loc, risks) in enumerate(top_zones, 1):
            avg_risk = sum(risks) / len(risks)
            risk_level = get_risk_level_color(avg_risk).split()[1]
            table_data.append((i, loc, f"{avg_risk:.1f}", risk_level))

        # Create DataFrame
        df_top = pd.DataFrame(table_data, columns=["Rank","Location", "Avg Risk", "Risk Level"])
        df_top.set_index("Rank", inplace=True)

        st.table(df_top)
    else:
        st.write("No detected locations to display.")

# --- Risk Distribution ---
st.subheader("📊 Risk Level Distribution")
fig, ax = plt.subplots(figsize=(4,2))  # smaller figure
ax.bar(risk_counts.keys(), risk_counts.values(), color=['green','yellow','orange','red'])
ax.set_xlabel("Risk Level")
ax.set_ylabel("Messages")
ax.tick_params(axis='x', rotation=0)
st.pyplot(fig, use_container_width=False)  # keep figure size fixed

# --- Top Drug Keywords ---
st.subheader("💊 Top Detected Drug Keywords")
top_drugs = drug_counter.most_common(20)
if top_drugs:
    df_drugs = pd.DataFrame(top_drugs, columns=["Drug","Count"])
    st.dataframe(df_drugs, use_container_width=True)
else:
    st.write("No drug keywords detected.")

# --- Drug Term Frequency ---
st.subheader("💊 Drug Term Frequency")
top_drugs = drug_counter.most_common(20)
df_drugs = pd.DataFrame(top_drugs, columns=["Drug", "Count"])
fig, ax = plt.subplots(figsize=(4,3))
ax.barh(df_drugs["Drug"], df_drugs["Count"], color="purple")
ax.invert_yaxis()
st.pyplot(fig, use_container_width=False)

# --- Flagged vs Clean Messages ---
st.subheader("🚨 Flagged vs Clean Messages")
flagged = len(df_filtered[df_filtered['Risk Score']>50])
clean = len(df_filtered) - flagged
fig, ax = plt.subplots(figsize=(3,3))  # smaller pie chart
ax.pie([flagged, clean], labels=["Flagged","Clean"], autopct='%1.1f%%', colors=["red","green"])
st.pyplot(fig, use_container_width=False)

# --- Drug Terms by Context ---
st.subheader("🧠 Drug Terms by Context")
context_counter = defaultdict(int)
# With this safer version:
for idx, row in df_filtered.iterrows():
    if row['Context'] and row['Drug Terms']:  # Check if not empty
        for ctx in row['Context']:
            for drug in row['Drug Terms']:
                context_counter[f"{drug}-{ctx}"] += 1
df_ctx = pd.DataFrame(context_counter.items(), columns=["Drug-Context","Count"]).sort_values("Count", ascending=False)
st.bar_chart(df_ctx.set_index("Drug-Context").head(10))

# --- Detection Method Breakdown ---
st.subheader("🧪 Detection Method Breakdown")
method_counts = df_filtered['Detection Method'].value_counts()
st.bar_chart(method_counts)

# --- Message Length vs Risk Score ---
st.subheader("✏️ Message Length vs Risk Score")
df_filtered['Length'] = df_filtered['Message'].apply(lambda x: len(str(x)))
fig, ax = plt.subplots(figsize=(4,2.5))  # smaller scatter
ax.scatter(df_filtered['Length'], df_filtered['Risk Score'], alpha=0.6)
ax.set_xlabel("Message Length")
ax.set_ylabel("Risk Score")
st.pyplot(fig, use_container_width=False)

# Sidebar Auto Refresh Control
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False

st.session_state.auto_refresh = st.sidebar.checkbox(
    "⏱️ Auto Refresh",
    value=st.session_state.auto_refresh
)

# --- Auto Refresh Logic ---
if st.session_state.auto_refresh:
    refresh_interval = 30  # seconds
    placeholder = st.empty()  # temporary container to force rerun
    while True:
        placeholder.text(f"Auto-refreshing every {refresh_interval} seconds...")
        time.sleep(refresh_interval)
        st.experimental_rerun()

# --- Footer ---
st.markdown("---")
st.markdown(
    "🚔 **WhatsApp Drug Detection Dashboard** | Built with Streamlit | "
    f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
