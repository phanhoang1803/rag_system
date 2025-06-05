# intelligent_rag_system/Dockerfile

# --- Stage 1: Build Environment & Dependencies ---
FROM python:3.12-slim AS builder

# Set working directory inside the container
WORKDIR /app

# Install system dependencies needed for some Python packages (e.g., SpaCy)
# and for building certain libraries (e.g., pypdf)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    default-libmysqlclient-dev \
    gcc \
    git \
    libffi-dev \
    libgmp-dev \
    libpq-dev \
    libssl-dev \
    zlib1g-dev \
    wget \
    # Clean up APT cache to reduce image size
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to leverage Docker cache
# If requirements.txt doesn't change, this layer won't rebuild
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Download SpaCy model (important for NERExtractor)
# This command needs to run AFTER pip install spacy
# RUN python -m spacy download en_core_web_sm && ls -R /usr/local/share/spacy

# --- Stage 2: Final Application Image ---
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy installed Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
# Copy SpaCy model data
# COPY --from=builder /usr/local/share/spacy /usr/local/share/spacy

# Copy the rest of your application code
# This includes src/, config/, run_ingestion.py, etc.
COPY . .

# Ensure the ChromaDB data directory exists and is writable
# This is where ChromaDB will persist its vector database.
# The docker-compose.yml will mount a host volume here for persistence.
RUN mkdir -p /app/chroma_data && chmod -R 777 /app/chroma_data

# Expose the ports that FastAPI and Streamlit will run on
# These ports are mapped to host ports in docker-compose.yml
# For FastAPI
EXPOSE 8000
# For Streamlit
EXPOSE 8501

# No default CMD is set here because `docker-compose.yml` explicitly defines
# the `command` for both `fastapi_app` and `streamlit_frontend` services.
# The `command` in docker-compose.yml will override any CMD instruction here.
# For example, FastAPI will run with `uvicorn src.main:app`, and Streamlit
# with `streamlit run src/frontend/app.py`.