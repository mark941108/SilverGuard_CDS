# Base Image: PyTorch with CUDA 12.1 (Official) - Ensures GPU compatibility
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Switch to root to install packages (the base image might default to non-root)
USER root

# Install system dependencies
# - espeak-ng: [CRITICAL] Required for offline TTS (pyttsx3)
# - libsndfile1: Required for torchaudio/librosa
# - fontconfig: Manage system fonts
# - wget: For downloading fonts
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    espeak-ng \
    libsndfile1 \
    fontconfig \
    libtesseract-dev \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# [Font Fix] Install Google Noto Sans TC (Traditional Chinese)
# This ensures matplotlib and PIL can render Chinese characters correctly without tofu blocks.
RUN mkdir -p /usr/share/fonts/opentype/noto && \
    wget -q https://github.com/google/fonts/raw/main/ofl/notosanstc/NotoSansTC-Regular.ttf -O /usr/share/fonts/opentype/noto/NotoSansTC-Regular.ttf && \
    wget -q https://github.com/google/fonts/raw/main/ofl/notosanstc/NotoSansTC-Bold.ttf -O /usr/share/fonts/opentype/noto/NotoSansTC-Bold.ttf && \
    fc-cache -fv

# Create a non-root user for security (SilverGuard Principle)
RUN useradd -m -u 1000 silverguard_user
USER silverguard_user
ENV HOME=/home/silverguard_user \
    PATH=/home/silverguard_user/.local/bin:$PATH

# Copy requirements and install python dependencies
COPY --chown=silverguard_user:silverguard_user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project code
COPY --chown=silverguard_user:silverguard_user . .

# Set Environment Variables
ENV OFFLINE_MODE=False \
    CUDA_VISIBLE_DEVICES=0 \
    MPLCONFIGDIR=/tmp/matplotlib

# Expose Gradio Port
EXPOSE 7860

# Default command: Launch Main App
# Using 'app.py' as it is the main entry point verified in README
CMD ["python", "app.py"]
