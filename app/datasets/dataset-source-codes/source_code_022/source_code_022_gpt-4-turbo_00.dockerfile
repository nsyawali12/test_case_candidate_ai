# Use an official Python runtime as a parent image in a multi-stage build process
FROM python:3.9-slim AS builder

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Use a second stage to create a small final image
FROM python:3.9-alpine

# Set the working directory to /app
WORKDIR /app

# Copy only the necessary artifacts from the builder stage
COPY --from=builder /app /app
COPY --from=builder /root/.cache /root/.cache

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]
