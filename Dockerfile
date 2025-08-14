FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

# Update to latest Python version
RUN conda update -n base -c defaults conda && \
    conda install python=3.9 -y

# Install system dependencies for matplotlib and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Update pip to latest version
RUN pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy all source code
COPY . .

# Create mount points for dataset and output
VOLUME ["/data", "/output"]

# Create dump.txt file in the output directory
RUN mkdir -p /output && echo "RangeViT Training Log" > /output/dump.txt

# Set the default command to use mounted volumes
CMD ["sh", "-c", "mkdir -p /output && echo 'RangeViT Training Started: $(date)' >> /output/dump.txt && python main.py config_kitti.yaml --data_root /data --save_path /output"]