# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# Install system dependencies required for Playwright and general tools
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers (Chromium only is usually sufficient)
RUN playwright install chromium

# Copy the rest of the application code
COPY . .

# Create a non-root user (Hugging Face Spaces requirement for security)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Expose the port (Hugging Face Spaces uses 7860 by default)
EXPOSE 7860

# Start the application
# Assuming your main entry point is in src/api/main.py and app instance is 'app'
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
