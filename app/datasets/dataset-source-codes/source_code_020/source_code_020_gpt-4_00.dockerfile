# Use an official lightweight Python image.
# "Alpine" version is usually more optimized for size and build speed.
FROM python:3.8-alpine

# Set the working directory
WORKDIR /app

# Add the Python script to the image
ADD hello.py /app

# The command to run the script when Docker image is run
CMD [ "python", "./hello.py" ]
