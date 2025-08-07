FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    # For PDF processing
    poppler-utils \
    # For python-magic
    libmagic1 \
    # For text processing
    libxml2-dev \
    libxslt1-dev \
    # For compiling C extensions
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir uvicorn==0.15.0 && \
    pip install --no-cache-dir -r requirements.txt && \
    # Install spacy model
    python -m spacy download en_core_web_sm

# Copy the application
COPY . .

# Expose the status port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/status || exit 1

CMD ["python", "worker.py"]