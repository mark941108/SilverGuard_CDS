# Base Image: PyTorch with CUDA support (T4 compatible)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
# espeak-ng: Required for SilverGuard TTS (Text-to-Speech)
# libsndfile1: Required for audio processing
RUN apt-get update && apt-get install -y \
    espeak-ng \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project code
COPY . .

# Set Environment Variables
# OFFLINE_MODE: Ensures the model runs without trying to phone home for datasets
ENV OFFLINE_MODE=False
ENV CUDA_VISIBLE_DEVICES=0

# Default command to run the inference service
# (Note: In a real deployment, this might be a FastAPI app or the Gradio interface)
CMD ["python", "HF_SPACE_APP.py"]
