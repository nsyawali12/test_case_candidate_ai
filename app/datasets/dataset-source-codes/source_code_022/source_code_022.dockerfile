# Stage 1: Build Stage
FROM python:3.8-slim AS builder

WORKDIR /app

# Install dependencies
RUN apt-get update \
    && apt-get install --no-install-recommends -y gcc \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir flask

# Stage 2: Production Stage
FROM python:3.8-slim

WORKDIR /app

# Copy only necessary files from the build stage
COPY --from=builder /usr/local/lib/python3.8/site-packages /usr/local/lib/python3.8/site-packages
COPY --from=builder /usr/local/bin/flask /usr/local/bin/flask

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

# Copy application code
COPY app.py .

# Expose the Flask port
EXPOSE 5000

# Start the Flask application
CMD ["flask", "run"]

app.py
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/answer')
def answer():
    result = (3 + 7 + 11) * 2
    return jsonify({'puzzle': result})

if __name__ == '__main__':
    app.run()