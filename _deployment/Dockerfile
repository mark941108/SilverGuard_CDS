# Base Image: PyTorch 2.6.0 with CUDA 12.4 (Performance Optimized for RTX 30/40/50 Series)
# Aligned with SilverGuard V12 Platinum & MedGemma requirements
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Set working directory
WORKDIR /app

# Switch to root for system installations
USER root

# Install system dependencies (Unified Package List)
# - espeak-ng: Required for pyttsx3 (Offline TTS Engine)
# - libsndfile1: Required for torchaudio / librosa
# - ffmpeg: Audio processing backend
# - fonts-noto-cjk: Core Chinese fonts for UI rendering
# - tesseract-ocr-chi-tra: Chinese OCR Language Data (Fallback Layer)
# - libgl1: Required for opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ffmpeg \
    espeak-ng \
    libsndfile1 \
    fontconfig \
    libgl1 \
    libtesseract-dev \
    tesseract-ocr \
    tesseract-ocr-chi-tra \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# [Font Cache Optimization] Pre-build font cache for faster startup
RUN fc-cache -fv

# [Permission Logic] Ensure /app is writable for temporary file generation
# Critical for non-root execution contexts (e.g., HF Spaces, OpenShift)
RUN chmod 777 /app

# Create a non-root user (Security Best Practice)
RUN useradd -m -u 1000 silverguard_user

# Pre-create critical directories with correct permissions
RUN mkdir -p /tmp/matplotlib && chmod 777 /tmp/matplotlib
RUN mkdir -p /home/silverguard_user/.cache && chmod -R 777 /home/silverguard_user/.cache

# Switch to non-root user
USER silverguard_user
ENV HOME=/home/silverguard_user \
    PATH=/home/silverguard_user/.local/bin:$PATH \
    MPLCONFIGDIR=/tmp/matplotlib \
    GRADIO_SERVER_NAME="0.0.0.0" \
    OFFLINE_MODE=True \
    CUDA_VISIBLE_DEVICES=0

# Copy requirements first to leverage Docker cache
COPY --chown=silverguard_user:silverguard_user requirements.txt .

# Install Python dependencies
# Utilizing PyTorch official wheel index for CUDA alignment reliability
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu124

# Copy entire project directory
COPY --chown=silverguard_user:silverguard_user . .

# [Injection Verification] Ensure critical data module exists
# The build will fail if medgemma_data.py is missing (Fail-Fast)
RUN python -c "import os; assert os.path.exists('medgemma_data.py'), 'CRITICAL: medgemma_data.py missing!'"

# Expose build info
LABEL org.opencontainers.image.source="https://github.com/mark941108/SilverGuard_CDS"
LABEL org.opencontainers.image.description="SilverGuard CDS V1.0 Impact Edition"

# Expose Gradio Port
EXPOSE 7860

# Launch Application
CMD ["python", "app.py"]
