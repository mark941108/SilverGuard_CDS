# Base Image: PyTorch 2.5.1 with CUDA 11.8 (Closest official to 2.6.0+cu118)
# We use 2.5.1 as absolute latest 2.6 official container might not be pushed yet to dockerhub
# But realistically for "V12 Platinum" we align as close as possible.
# Let's use nvidia/cuda base and install py2.6 manually or use slightly older torch base.
# Actually, KAGGLE_BOOTSTRAP installs 2.6.0 via pip. Best to use a solid Python 3.10/3.11 base with CUDA.
# However, to keep it simple, we will stick to a standard pytorch base and upgrade inside if needed,
# or just use the official image that supports our needs.
# Pytorch 2.5.1 is robust. Let's use that as base and let pip upgrade to 2.6.0 if requirements.txt demands it.
FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime

# Set working directory
WORKDIR /app

# Switch to root to install packages
USER root

# Install system dependencies (Aligned with KAGGLE_BOOTSTRAP.py)
# - espeak-ng: Required for pyttsx3 (Offline TTS)
# - libsndfile1: Required for torchaudio
# - ffmpeg: Audio processing
# - fonts-noto-cjk: [CRITICAL] Core Chinese fonts for SilverGuard UI
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    espeak-ng \
    libsndfile1 \
    fontconfig \
    libtesseract-dev \
    tesseract-ocr \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# [Font Cache Update]
RUN fc-cache -fv

# Create a non-root user for security (SilverGuard Principle)
RUN useradd -m -u 1000 silverguard_user
USER silverguard_user
ENV HOME=/home/silverguard_user \
    PATH=/home/silverguard_user/.local/bin:$PATH

# Copy requirements and install python dependencies
COPY --chown=silverguard_user:silverguard_user requirements.txt .
# Combine pip install to minimize layers and force index-url for cu118 if needed
RUN pip install --no-cache-dir -r requirements.txt --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple



# Copy the entire project code
COPY --chown=silverguard_user:silverguard_user . .

# Set Environment Variables
ENV OFFLINE_MODE=False \
    CUDA_VISIBLE_DEVICES=0 \
    MPLCONFIGDIR=/tmp/matplotlib \
    GRADIO_SERVER_NAME="0.0.0.0"

# Expose Gradio Port
EXPOSE 7860

# Default command: Launch Main App
CMD ["python", "app.py"]
