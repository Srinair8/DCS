#drug_audio_classifier/Dockerfile
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 curl && \
    rm -rf /var/lib/apt/lists/* && pip install --no-cache-dir -r requirements.txt

COPY . /app
RUN adduser --disabled-password --gecos '' appuser && chown -R appuser /app
USER appuser

EXPOSE 8503
CMD ["streamlit","run","app.py","--server.address=0.0.0.0","--server.port=8503","--server.baseUrlPath=/audio","--browser.gatherUsageStats=false","--server.enableCORS=false" , "--server.headless=true"]

