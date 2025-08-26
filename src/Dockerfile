# main_app/Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app

RUN adduser --disabled-password --gecos '' appuser && chown -R appuser /app
USER appuser

EXPOSE 8501
CMD ["streamlit","run","src/streamlit_app.py","--server.address=0.0.0.0","--server.port=8501","--server.baseUrlPath=/","--browser.gatherUsageStats=false","--server.enableCORS=false" , "--server.headless=true"]
