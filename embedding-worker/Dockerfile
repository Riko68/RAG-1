FROM python:3.9-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

# Build sentencepiece from source
RUN pip install --no-cache-dir cmake && \
    git clone --depth 1 --branch v0.1.98 https://github.com/google/sentencepiece.git /tmp/sentencepiece && \
    mkdir -p /tmp/sentencepiece/build && \
    cd /tmp/sentencepiece/build && \
    cmake .. && \
    make -j $(nproc) && \
    make install && \
    ldconfig

# Now create the final image
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
    # For sentencepiece
    cmake \
    pkg-config \
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

# Copy the built sentencepiece
COPY --from=builder /usr/local/lib/ /usr/local/lib/
COPY --from=builder /usr/local/include/ /usr/local/include/

# Install Python packages
COPY requirements.txt .

# Install requirements
RUN pip install --no-cache-dir uvicorn==0.15.0 && \
    pip install --no-cache-dir -r requirements.txt && \
    # Install spacy models
    python -m spacy download en_core_web_md && \
    python -m spacy download fr_core_news_md && \
    python -m spacy download de_core_news_md

# Copy the application
COPY . .

# Expose the status port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/status || exit 1

CMD ["python", "worker.py"]