# Stage 1: Build
FROM python:3.8-slim as builder

# Set up a virtual environment to install packages
WORKDIR /app
RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Stage 2: Final image
FROM python:3.8-slim

# Copy virtual environment from builder
COPY --from=builder /app/venv /app/venv

# Set environment to use virtual env
ENV PATH="/app/venv/bin:$PATH"

# Copy application code
WORKDIR /app
COPY app.py .

# Expose the application port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
