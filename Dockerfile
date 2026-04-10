FROM python:3.11-slim

# Install system dependencies for LightGBM/XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Train models during build
RUN python -m ml.train

# Expose port
EXPOSE 8080

# Start with gunicorn
CMD gunicorn app:app --bind 0.0.0.0:${PORT:-8080} --workers 2 --timeout 300
