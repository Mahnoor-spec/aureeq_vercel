# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (needed for certain python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY model_training/scripts/requirements.txt .

# Upgrade pip and install packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend code and data
# We need to preserve the relative structure for data loading
COPY model_training /app/model_training
COPY data /app/data

# Set working directory to where server.py is
WORKDIR /app/model_training/scripts

# Make port 8001 available to the world outside this container
EXPOSE 8001

# Run server.py when the container launches
CMD ["python", "server.py"]
