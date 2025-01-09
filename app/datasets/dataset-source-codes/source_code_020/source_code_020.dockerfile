FROM python:3.8
WORKDIR /app

COPY helloworld.py .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "helloworld.py"]


## PYTHON PROGRAM
helloworld.py
print("Hello, World!")


## BUILD COMMAND
docker build -t "python:helloworld" .
docker run -itd --name python python:helloworld