# Step 1: Choose a smaller and official Python base image for build stage
FROM python:3.8-slim-buster as build

WORKDIR /app

# Set environment variables
# PYTHONDONTWRITEBYTECODE prevents Python from writing pyc files to disc
# PYTHONUNBUFFERED ensures stdout and stderr are not buffered
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install dependencies
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY . /app

# Step 2: Produce a smaller image for the deployment stage
FROM python:3.8-alpine3.12 

WORKDIR /app

# Copy the necessary files from the build image
COPY --from=build /root/.local /root/.local
COPY --from=build /app /app

# Place any extras you want here

# Make Python packages installed globally in the build stage available in the second stage
ENV PATH=/root/.local:$PATH
    
# Define the Flask app environment variable
ENV FLASK_APP=main.py

# Expose the Flask port
EXPOSE 5000

# Set the default command for the container
CMD ["flask", "run", "--host=0.0.0.0"]
