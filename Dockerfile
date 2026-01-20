# Stage 1: Build stage
FROM python:3.7-slim AS builder

# Set environment variables to prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a virtual environment or local path
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Final Run stage
FROM python:3.7-slim

WORKDIR /app

# Copy only the installed python packages from the builder stage
COPY --from=builder /root/.local /root/.local
COPY . /app

# Update PATH to include the user-level bin directory
ENV PATH=/root/.local/bin:$PATH

# Expose the port your app runs on
EXPOSE 5050

# Use Gunicorn for production as it's included in your requirements
# This is more robust than the Flask development server used in main_app.py
CMD ["gunicorn", "--bind", "0.0.0.0:5050", "main_app:app"]
