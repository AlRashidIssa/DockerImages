# Use an existing base image
FROM ubuntu:latest

# Update package repositories and install dependencies
RUN apt-get update 

# Set the working directory
WORKDIR /app

# Copy your application code into the image
COPY . .

# Install any Python dependencies
RUN pip install -r requirements.txt


# Define the command to run your application
CMD ["python3", "FaceDetection.py"]
