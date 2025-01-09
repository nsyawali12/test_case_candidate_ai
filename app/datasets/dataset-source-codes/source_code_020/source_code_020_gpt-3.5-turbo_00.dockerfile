# Use a slim Python base image for optimization
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the Python script into the container
COPY hello.py /app/hello.py

# Define the command to run the Python script
CMD ["python", "/app/hello.py"]
