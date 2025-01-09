# Use an official Python runtime as a parent image
FROM python:3.8-slim as builder

# Set the working directory in the builder stage container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any necessary dependencies
RUN pip install --upgrade pip && \
    pip install flask gunicorn

# For the production environment
FROM python:3.8-slim as runner

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Copy only the dependencies installation from the 1st stage image
COPY --from=builder /usr/local/lib/python3.8/site-packages /usr/local/lib/python3.8/site-packages

# Set the working our directory in the image
WORKDIR /app

# Copy our source code
COPY . /app

# Allows port 5000 to be publicly available
EXPOSE 5000

# Command to start the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
