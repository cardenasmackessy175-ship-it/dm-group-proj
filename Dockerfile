# Use slim Python image to minimize size
# Using Alibaba Cloud mirror for faster and more reliable access
FROM registry.cn-hangzhou.aliyuncs.com/library/python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Copy model files
COPY Subtask2.1-sentiment_analysis/sentiment_model_subtask2.1.joblib ./Subtask2.1-sentiment_analysis/
COPY subtask2.2_topic_model/topic_classification_model_subtask2.2.joblib ./subtask2.2_topic_model/

# Expose port 5724
EXPOSE 5724

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Use gunicorn for production with limited workers to control memory
# --workers=2: Use 2 worker processes to stay within 900MB memory limit
# --threads=2: Use 2 threads per worker
# --timeout=120: Increase timeout for model loading
# --worker-class=sync: Use synchronous workers (lower memory)
CMD ["gunicorn", "--bind", "0.0.0.0:5724", "--workers", "2", "--threads", "2", "--timeout", "120", "--worker-class", "sync", "app:app"]
