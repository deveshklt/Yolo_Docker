FROM python:3.12-slim

WORKDIR /app

# Install system dependencies required by OpenCV and Git LFS
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Initialize Git LFS (optional, you can remove if not pulling LFS inside container)
RUN git lfs install

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application and YOLOv5 folder (including LFS-tracked .pt files)
COPY . .

# Remove the failing line
# RUN git lfs pull

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
