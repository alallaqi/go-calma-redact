FROM python:3.11-slim

# System dependencies: Tesseract OCR with language packs + poppler for pdf2image
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-deu \
    tesseract-ocr-fra \
    tesseract-ocr-ita \
    tesseract-ocr-eng \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download NER model (~680 MB) — baked into image so first run is instant
RUN python -c "from transformers import pipeline; pipeline('ner', model='Davlan/bert-base-multilingual-cased-ner-hrl', aggregation_strategy='simple')"

# Pre-download Flan-T5 summariser (~77 MB)
RUN python -c "from transformers import pipeline; pipeline('text2text-generation', model='MBZUAI/LaMini-Flan-T5-77M')"

# Copy application source
COPY app.py .
COPY ui/ ui/
COPY gocalma/ gocalma/
COPY assets/ assets/
COPY .streamlit/ .streamlit/

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false", "--server.enableCORS=false"]
