#whatsapp_drug_detector/Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app

RUN adduser --disabled-password --gecos '' appuser && chown -R appuser /app
USER appuser

EXPOSE 8502
CMD ["streamlit","run","app.py","--server.address=0.0.0.0","--server.port=8502","--server.baseUrlPath=/whatsapp","--browser.gatherUsageStats=false","--server.enableCORS=false" , "--server.headless=true"]
