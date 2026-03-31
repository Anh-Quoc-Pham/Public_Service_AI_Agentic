FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy Rasa configuration and data
COPY config.yml .
COPY domain.yml .
COPY data/ ./data/
COPY actions/ ./actions/

# Create necessary directories
RUN mkdir -p /app/models /app/logs

# Expose port
EXPOSE 5005

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5005/status || exit 1

# Run Rasa
CMD ["rasa", "run", "--enable-api", "--cors", "*", "--port", "5005"] 