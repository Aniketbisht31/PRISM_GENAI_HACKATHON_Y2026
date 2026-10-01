FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so it's baked into the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code
COPY src/ ./src/
COPY prompts/ ./prompts/
COPY data/corpus/ ./data/corpus/
COPY data/eval/ ./data/eval/
COPY data/chroma_index/ ./data/chroma_index/
COPY eval/ ./eval/
COPY tests/ ./tests/

# Index the corpus at build time
RUN python -c "from src.retrieval.retriever import HybridRetriever; r = HybridRetriever(); r.index_corpus('data/corpus/workshop_planning_corpus.json')"

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
