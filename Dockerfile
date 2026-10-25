FROM python:3.11-slim

# Prevent python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    HOME=/home/user

# Install system dependencies (build-essential, curl, git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (required for Hugging Face Spaces & security best practices)
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# Pre-install CPU-only PyTorch (reduces image size by >1.5GB compared to CUDA wheel)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code, data, and trained model artifacts
COPY --chown=user:user . .

# Pre-download embedding model and pre-build Chroma index so cold starts are instant
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" && \
    python -c "from src.retrieval.indexer import ChromaRetriever; r = ChromaRetriever(); print(f'Chroma index ready with {r.count()} passages.')"

# Switch to non-root user
USER user

# Expose standard port (7860 for Hugging Face Spaces, Render dynamically injects $PORT)
EXPOSE 7860

# Launch server binding to $PORT (works on Hugging Face, Render, Koyeb, Railway)
CMD ["sh", "-c", "uvicorn src.api.server:app --host 0.0.0.0 --port ${PORT:-7860}"]
