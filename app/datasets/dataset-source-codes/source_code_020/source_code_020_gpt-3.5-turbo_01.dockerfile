# Use a lightweight base image for faster build speed and smaller final image size
FROM python:3.8-alpine

# Set working directory inside the container
WORKDIR /app

# Copy the Python script into the container
COPY hello.py .

# Define the command to run the Python script
CMD ["python", "hello.py"]
