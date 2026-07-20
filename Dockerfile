# ==============================================================================
# Multi-Agent AI Data Analyst - Multi-Stage Production Dockerfile
# ==============================================================================

# --- Stage 1: Build Dependencies ---
FROM python:3.12-slim-bookworm AS builder

# Set build-time variables and environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# Install build dependencies (compilers and library headers for C-extensions)
# WeasyPrint requires several shared libraries (Pango, Cairo, etc.)
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    libffi-dev \
    libssl-dev \
    libcairo2-dev \
    libpango1.0-dev \
    libgdk-pixbuf2.0-dev \
    shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy dependency specifications
COPY requirements.txt .

# Create virtual environment and install packages
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# --- Stage 2: Final Runtime ---
FROM python:3.12-slim-bookworm AS runner

# Environment settings
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    WORKSPACE_DIR=/app/workspace \
    PORT=8501

# Install runtime dependencies for ML/plotting libraries and WeasyPrint PDF generation
RUN apt-get update && apt-get install --no-install-recommends -y \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf2.0-0 \
    shared-mime-info \
    fonts-dejavu \
    fonts-liberation \
    git \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy configuration files and templates
COPY pyproject.toml README.md LICENSE ./

# Copy directory structure
COPY app/ ./app/
COPY src/ ./src/
COPY workspace/ ./workspace/
COPY tests/ ./tests/
COPY docs/ ./docs/
COPY scripts/ ./scripts/

# Create a non-privileged user and group for runtime security
RUN groupadd -g 10001 appuser && \
    useradd -u 10001 -g appuser -d /app -s /sbin/nologin appuser && \
    chown -R appuser:appuser /app

# Switch to the non-privileged user
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Healthcheck to verify Streamlit server status
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default runtime entry point (run Streamlit web dashboard)
CMD ["streamlit", "run", "app/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
