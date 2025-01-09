# Use an official Python runtime as the base image to optimize the image size
FROM python:3.9-slim-buster 

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app (this will include the Python script)
ADD . /app

# Define environment variable to prevent Python from writing *.pyc files to disc (optional)
ENV PYTHONDONTWRITEBYTECODE 1

# Define environment variable to prevent Python from buffering stdout and stderr (optional)
ENV PYTHONUNBUFFERED 1

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Define the script (we assume it's called script.py) that should be run when the container is launched
CMD ["python", "./script.py"]
