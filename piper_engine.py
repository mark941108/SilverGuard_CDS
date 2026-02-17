# piper_engine.py
# 🚀 High-speed Offline Neural TTS for SilverGuard CDS (Windows/Linux)to-download of Piper binary and ONNX models for ID and VI.

import os
import sys
import subprocess
import zipfile
import requests
from pathlib import Path

# Configuration
PIPER_DIR = Path("piper_tts")
PIPER_EXE = PIPER_DIR / "piper" / "piper.exe"
MODELS_DIR = PIPER_DIR / "models"

# URLs
PIPER_ZIP_URL = "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip"

LANGUAGE_MODELS = {
    "id": {
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/id/id_ID/news_tts/medium/id_ID-news_tts-medium.onnx",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/id/id_ID/news_tts/medium/id_ID-news_tts-medium.onnx.json",
        "file_name": "id_ID-news_tts-medium.onnx"
    },
    "vi": {
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx.json",
        "file_name": "vi_VN-vais1000-medium.onnx"
    }
}
# NOTE: ZH-TW and EN use native system TTS (SAPI5/eSpeak) for offline mode.
# Piper models for these languages are not currently bundled to save bandwidth.

def ensure_piper_assets(lang="id"):
    """確保 Piper 執行檔與對應語言模型存在，若無則自動下載"""
    PIPER_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    # 1. Check & Download Piper Binary
    if not PIPER_EXE.exists():
        print(f"⬇️ Downloading Piper standalone binary tool...")
        zip_path = PIPER_DIR / "piper_windows.zip"
        try:
            response = requests.get(PIPER_ZIP_URL, timeout=30)
            with open(zip_path, "wb") as f:
                f.write(response.content)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(PIPER_DIR)
            
            os.remove(zip_path) # Cleanup
            print("✅ Piper binary extracted.")
        except Exception as e:
            print(f"❌ Failed to download Piper: {e}")
            return False

    # 2. Check & Download Language Model
    if lang not in LANGUAGE_MODELS:
        return False
        
    model_cfg = LANGUAGE_MODELS[lang]
    onnx_path = MODELS_DIR / model_cfg["file_name"]
    json_path = MODELS_DIR / (model_cfg["file_name"] + ".json")

    # [Fix] Both ONNX and JSON must exist
    if not onnx_path.exists() or not json_path.exists():
        print(f"⬇️ Downloading offline voice model for '{lang}'...")
        try:
            # Download ONNX
            r = requests.get(model_cfg["onnx"], timeout=60)
            with open(onnx_path, "wb") as f: f.write(r.content)
            # Download JSON
            r = requests.get(model_cfg["json"], timeout=30)
            with open(json_path, "wb") as f: f.write(r.content)
            print(f"✅ Voice model for '{lang}' ready.")
        except Exception as e:
            print(f"❌ Failed to download models for {lang}: {e}")
            return False

    return True

def generate_speech_offline(text, output_file, lang="id"):
    """使用 Piper 執行檔進行離線語音合成"""
    if not ensure_piper_assets(lang):
        return False

    model_path = MODELS_DIR / LANGUAGE_MODELS[lang]["file_name"]
    
    try:
        # Command: echo "text" | piper.exe -m model.onnx -f output.wav
        # We use absolute paths to be safe
        cmd = [
            str(PIPER_EXE.absolute()),
            "--model", str(model_path.absolute()),
            "--output_file", str(Path(output_file).absolute())
        ]
        
        print(f"🎤 [Piper Offline] Generating logic for '{lang}'...")
        # Use subprocess with input pipe
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        
        try:
            stdout, stderr = process.communicate(input=text, timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            print(f"❌ [Piper] Process timed out (30s) for '{lang}'")
            return False
        
        if process.returncode == 0 and os.path.exists(output_file):
            return True
        else:
            print(f"❌ Piper Execution Error: {stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Piper Interface Fatal Error: {e}")
        return False

if __name__ == "__main__":
    # Test
    test_file = "test_offline.wav"
    if generate_speech_offline("Halo, ini adalah suara offline Piper.", test_file, lang="id"):
        print(f"🎉 Test Success! Audio saved to {test_file}")
    else:
        print("❌ Test Failed.")
