FROM python:3.11-slim

# poppler-utils: required by pdf2image for PDF rendering
# No Tesseract — Surya is the primary OCR engine (transformer-based, no language packs needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# CPU-only PyTorch first (~500 MB vs ~2 GB for the full CUDA wheel)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download Surya OCR models — baked in so first upload is instant
RUN python -c "\
from surya.model.detection.model import load_model as d, load_processor as dp; \
from surya.model.recognition.model import load_model as r; \
from surya.model.recognition.processor import load_processor as rp; \
d(); dp(); r(); rp()"

# Pre-download multilingual NER model (~680 MB)
RUN python -c "\
from transformers import pipeline; \
pipeline('ner', model='Davlan/bert-base-multilingual-cased-ner-hrl', aggregation_strategy='simple')"

# Pre-download privacy risk summariser (~77 MB)
RUN python -c "\
from transformers import pipeline; \
pipeline('text2text-generation', model='MBZUAI/LaMini-Flan-T5-77M')"

# Pre-download LLM for entity verification (~1 GB in float16)
# Used in-process — no Ollama daemon required
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForCausalLM; \
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct'); \
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', torch_dtype='auto')"

# Copy application source
COPY app.py .
COPY ui/ ui/
COPY gocalma/ gocalma/
COPY assets/ assets/
COPY .streamlit/ .streamlit/

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.enableXsrfProtection=false", \
     "--server.enableCORS=false"]
