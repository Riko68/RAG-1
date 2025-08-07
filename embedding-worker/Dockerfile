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
    # For sentencepiece and other build tools
    cmake \
    pkg-config \
    git \
    # For language detection
    libicu-dev \
    # For Python packages
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Update pip and setuptools first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python packages
COPY requirements.txt .

# Install requirements with specific versions and trusted hosts
RUN pip install --no-cache-dir \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    uvicorn==0.15.0 && \
    pip install --no-cache-dir \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    -r requirements.txt

# Install spacy models from direct URLs in requirements.txt
# The models are already downloaded as part of the requirements installation
# We just need to link them
RUN python -m spacy link --force en_core_web_md en_core_web_md && \
    python -m spacy link --force fr_core_news_md fr_core_news_md && \
    python -m spacy link --force de_core_news_md de_core_news_md && \
    python -m spacy link --force en_core_web_sm en_core_web_sm

# Copy the application
COPY . .

# Expose the status port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/status || exit 1

CMD ["python", "worker.py"]