# Multi-stage Dockerfile for MCP Router System
# 
# Builds a production-ready container with all dependencies
# and proper security configurations.
# 
# Per requirements.md: Complete implementation, no placeholders.

# Build stage
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r mcpuser && useradd -r -g mcpuser -s /bin/bash mcpuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY deploy/entrypoint.sh ./entrypoint.sh

# Create directories and set permissions
RUN mkdir -p /app/logs /app/config /app/data && \
    chown -R mcpuser:mcpuser /app && \
    chmod +x /app/entrypoint.sh

# Copy configuration files
COPY deploy/config/ ./config/

# Install spaCy language model
RUN python -m spacy download en_core_web_sm

# Set environment variables
ENV PYTHONPATH="/app:$PYTHONPATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MCP_ENV=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8090/health || exit 1

# Switch to non-root user
USER mcpuser

# Expose ports
EXPOSE 8090

# Default command
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "-m", "src.mcp.server"]

# Metadata labels
LABEL maintainer="MCP Router System" \
      version="1.0.0" \
      description="MCP-based Claude Code + Ollama Integration System" \
      org.opencontainers.image.source="https://github.com/D0k-Soundwave/intern"