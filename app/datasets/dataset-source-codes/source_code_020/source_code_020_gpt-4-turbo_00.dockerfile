# Use an official lightweight Python image.
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the Python script into the container
COPY hello.py .

# Run the Python script when the container launches
CMD ["python", "./hello.py"]
