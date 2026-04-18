# Base image
FROM python:3.11-slim

# Install system dependencies required to build Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
        libffi-dev \
        python3-dev \
        pkg-config \
        git \
        curl \
        wget && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Copy rest of your bot code
COPY . .

# Default command
CMD ["bash"]
# 1. Base image
FROM python:3.11-slim

# 2. Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
        git \
        libffi-dev \
        python3-dev \
        pkg-config && \
    rm -rf /var/lib/apt/lists/*

# 3. Set working directory
WORKDIR /app

# 4. Copy requirements
COPY requirements.txt .

# 5. Install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# 6. Copy the rest of the bot code
COPY . .

# 7. Default command (can be overwritten)
CMD ["bash"]
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
        git \
        libffi-dev \
        python3-dev \
        pkg-config && \
    rm -rf /var/lib/apt/lists/*
FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Copy the rest of the bot scripts
COPY . .

# Default command: start script
CMD ["bash", "start.sh"]
