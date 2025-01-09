# Use an official Python image as the base image
FROM python:3.8-slim AS builder

# Set the working directory
WORKDIR /app

# Copy the Python requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python application code
COPY app/ .

# Build a small production image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy installed dependencies
COPY --from=builder /root/.cache /root/.cache

# Copy the Python application code
COPY --from=builder /app /app

# Expose port 5000
EXPOSE 5000

# Define the command to run the Flask app
CMD ["python", "app.py"]
