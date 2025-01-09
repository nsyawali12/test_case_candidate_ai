# First stage - Build the Python application
FROM python:3.8 AS builder

WORKDIR /app

# Copy only the requirements file to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install Flask

# Copy the rest of the application code
COPY . .

# Second stage - Create the optimized Docker image
FROM python:3.8-slim

WORKDIR /app

# Copy the built Python dependencies from the first stage
COPY --from=builder /usr/local/lib/python3.8/site-packages /usr/local/lib/python3.8/site-packages

# Copy the application code
COPY . .

# Expose the port
EXPOSE 5000

# Command to run the Flask web application
CMD ["python", "app.py"]
