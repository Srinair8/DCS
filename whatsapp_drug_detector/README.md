# WhatsApp Drug Chat Detection System

## 📝 Overview
The **WhatsApp Drug Chat Detection System** is an AI-powered tool designed to assist law enforcement in detecting drug-related conversations on WhatsApp in real-time. It uses **natural language processing (NLP)** to classify messages, extract locations, assign risk scores, and generate alerts, all without relying on external police databases like CCTNS. The system includes a web dashboard for analysis and real-time WhatsApp integration for live monitoring.

### Key Features
- **Message Classification**: Uses a BERT-based transformer model to classify messages as **suspicious** (drug-related) or **clean**, with a confidence score.
- **Location Extraction**: Employs spaCy’s Named Entity Recognition (NER) to identify Bengaluru hotspots (e.g., BTM, Majestic) and map them with risk weights.
- **Risk Scoring**: Combines model confidence and location risk to assign a score and label (High, Medium, Low).
- **Web Dashboard**: A Streamlit app (`app.py`) for analyzing messages, displaying heatmaps, and exporting reports as CSV.
- **WhatsApp Integration**: A Flask webhook (`whatsapp_listener.py`) connects to the Twilio WhatsApp API for real-time message processing and alerts.
- **Visualization**: Folium-based heatmaps and top 10 risky areas displayed on the dashboard.

## 🛠️ Tech Stack
| Component               | Purpose                                                                 |
|------------------------|-------------------------------------------------------------------------|
| **Python**             | Core programming language.                                              |
| **Streamlit**          | Interactive web dashboard for message analysis and visualization.        |
| **Flask**              | Web server for handling WhatsApp webhook.                               |
| **BERT (Transformers)**| AI model for classifying drug-related messages.                         |
| **spaCy (NER)**        | Extracts location names from messages.                                  |
| **Twilio WhatsApp API**| Enables real-time WhatsApp message sending/receiving.                   |
| **Ngrok**              | Creates secure public URLs for local testing.                           |
| **Pandas**             | Data processing and CSV export.                                        |
| **Folium/MarkerCluster**| Generates heatmaps and visualizations of risky areas.                   |

## 📂 Project Structure
```
whatsapp-drug-chat-detection/
├── app.py                  # Streamlit app for web dashboard
├── whatsapp_listener.py    # Flask app for WhatsApp webhook
├── predict_local.py        # BERT-based message classification script
├── data/                   # Directory for drug slang dictionary and hotspot data
│   ├── slang_dict.json     # Indian drug slang terms
│   ├── hotspots.json       # Bengaluru hotspots with coordinates and risk weights
├── requirements.txt        # Python dependencies
├── README.md               # This file
```

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8+
- A Twilio account with WhatsApp Sandbox enabled
- Ngrok for local testing
- Git for cloning the repository

### Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-repo>/whatsapp-drug-chat-detection.git
   cd whatsapp-drug-chat-detection
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Ensure `requirements.txt` includes:
   ```
   streamlit
   flask
   transformers
   spacy
   twilio
   pandas
   folium
   ```

3. **Download spaCy model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Set up Twilio WhatsApp API**:
   - Create a Twilio account and enable the WhatsApp Sandbox.
   - Note your **Account SID**, **Auth Token**, and **WhatsApp Sandbox Number**.
   - Configure the sandbox to point to your Flask webhook (see below).

5. **Prepare data files**:
   - Place `slang_dict.json` (drug slang terms) and `hotspots.json` (Bengaluru locations with coordinates and risk weights) in the `data/` directory.

### Running the System
1. **Start the Flask webhook** (for WhatsApp integration):
   ```bash
   python whatsapp_listener.py
   ```
   - Use Ngrok to expose the local server:
     ```bash
     ngrok http 5000
     ```
   - Copy the Ngrok URL (e.g., `https://<ngrok-id>.ngrok.io`) and set it as the Twilio WhatsApp webhook URL in the Twilio console.

2. **Start the Streamlit dashboard**:
   ```bash
   streamlit run app.py
   ```
   - Access the dashboard at `http://localhost:8501`.

3. **Test the system**:
   - Send a test message to the Twilio WhatsApp Sandbox number (e.g., “Got maal in Indiranagar tonight”).
   - Check the Flask console for real-time analysis and WhatsApp for the reply.
   - View results, heatmaps, and top risky areas on the Streamlit dashboard.

## 🔄 System Flow
1. **Input**: User sends a WhatsApp message.
2. **Processing**:
   - Twilio forwards the message to the Flask webhook (`whatsapp_listener.py`).
   - BERT model (`predict_local.py`) classifies the message.
   - spaCy NER extracts locations.
   - Risk score is calculated based on confidence and location weight.
3. **Output**:
   - Flask sends a WhatsApp reply with analysis (e.g., suspicious flag, location, risk score).
   - Streamlit dashboard displays results, heatmap, and top 10 risky areas.
   - Results can be exported as CSV.

## 📌 Current Status
- **Fully Functional**:
  - Real-time message classification and location extraction.
  - Risk scoring with High/Medium/Low labels.
  - Streamlit dashboard with heatmaps and CSV export.
  - Twilio WhatsApp API integration for live alerts.
- **Customized**: Tailored for Karnataka police with local slang and Bengaluru hotspots.
- **Tested**: End-to-end pipeline validated with test messages (e.g., “Stuff in Koramangala” → Confidence: 97.2%, Risk Score: 132, High risk).

## 🔜 Next Steps
- Add a **database** to store flagged messages with timestamps.
- Build an **admin panel** for law enforcement to review chats.
- Support **images, videos, and multi-language** (e.g., Kannada, Hindi) analysis.
- Deploy to **cloud** (AWS/Heroku) for scalability.
- Implement **alerts** (email/SMS/WhatsApp) for high-risk messages.
- Detect **school/college mentions** and **vulnerable groups**.
- Add **dark web phrase** detection.

## 🛡️ Law Enforcement Use
- **Real-time**: Immediate alerts and heatmaps for quick action.
- **Standalone**: No dependency on CCTNS or external databases.
- **Extensible**: Can be adapted for other crimes (e.g., fraud, trafficking).

## 📚 Example
**Input Message**: “Got stuff in Koramangala tonight”  
**WhatsApp Reply**:
```
🚨 Suspicious message detected!
📍 Location: Koramangala
🔢 Confidence: 97.2%
⚠️ Risk Score: 132
🛡️ Action recommended.
```
**Dashboard Output**: Table with highlighted terms, location, risk score, and heatmap marking Koramangala as a red zone.
