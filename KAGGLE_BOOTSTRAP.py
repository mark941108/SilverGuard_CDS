"""
================================================================================
🏥 AI Pharmacist Guardian - Kaggle Bootstrap (V10.0 Clean Slate)
================================================================================
📋 戰略總監驗證 (Verified Strategy):
   1. [NUKE] 暴力移除所有 PyTorch 與 HuggingFace 相關套件，清除殘留。
   2. [PAVE] 從 PyPI 下載官方驗證過的「黃金三角」版本。
   3. [VERIFY] 安裝後立即執行自我檢測，確保 import 成功。
================================================================================
"""

# %%
# ============================================================================
# STEP 0: 環境重置與認證
# ============================================================================
import os
import sys
import subprocess
from kaggle_secrets import UserSecretsClient

print("=" * 80)
print("🏥 AI Pharmacist Guardian - Bootstrap (V10.0 Clean Slate)")
print("=" * 80)

# 1. 讀取金鑰
user_secrets = UserSecretsClient()
print("\n[1/6] 讀取認證金鑰...")
try:
    gh_token = user_secrets.get_secret("GITHUB_TOKEN")
    hf_token = user_secrets.get_secret("HUGGINGFACE_TOKEN")
    print("   ✅ 金鑰讀取成功")
except:
    print("   ❌ 金鑰未設定！請去 Add-ons > Secrets 設定")
    gh_token = ""
    hf_token = ""

# %%
# ============================================================================
# STEP 1: 下載 Repository
# ============================================================================
print("\n[2/6] 下載 SilverGuard Repository...")
!rm -rf SilverGuard medgemma_training_data_v5
repo_url = f"https://{gh_token}@github.com/mark941108/SilverGuard.git"
!git clone --depth 1 {repo_url}
%cd SilverGuard
print("   ✅ Repository 下載完成")

# %%
# ============================================================================
# STEP 2: 自動熱修復 (Hotfix Patch)
# ============================================================================
print("\n[3/6] 應用代碼熱修復...")
patch_code = """
    "QD_breakfast_after": {"code": "QD-PC", "zh": "每日1次，早餐後服用", "detail": "每日早餐後30分鐘服用"},
    "QD_meals_with": {"code": "QD-M", "zh": "每日1次，隨餐服用", "detail": "請於用餐時一併服用以增加吸收"},
"""
target_file = "SilverGuard_Impact_Research_V8.py"
try:
    with open(target_file, "r", encoding="utf-8") as f:
        content = f.read()
    if '"QD_meals_with":' not in content:
        anchor = '"QD_breakfast_after": {"code": "QD-PC", "zh": "每日1次，早餐後服用", "detail": "每日早餐後30分鐘服用"},'
        if anchor in content:
            new_content = content.replace(anchor, patch_code.strip())
            with open(target_file, "w", encoding="utf-8") as f:
                f.write(new_content)
            print("   ✅ 熱修復成功")
    else:
        print("   ✅ 代碼已包含修復")
except Exception as e:
    print(f"   ⚠️ 熱修復跳過: {e}")

# %%
# ============================================================================
# STEP 3: 暴力清除舊環境 (The Nuke)
# ============================================================================
print("\n[4/6] 正在清理衝突套件 (這可能需要 1 分鐘)...")
# 強制移除所有可能衝突的套件
!pip uninstall -y torch torchvision torchaudio transformers huggingface_hub sentence-transformers accelerate peft bitsandbytes gradio

# %%
# ============================================================================
# STEP 4: 乾淨安裝 (The Pave) - 黃金版本矩陣
# ============================================================================
print("\n[5/6] 安裝黃金版本組合...")

# 1. 系統依賴
!apt-get update -y && apt-get install -y libespeak1 libsndfile1 ffmpeg

# 2. PyTorch 生態系 (嚴格鎖定版本)
# Torch 2.5.1 是目前最穩定的 CUDA 12 版本
print("   ⬇️ 安裝 PyTorch 2.5.1 Ecosystem...")
!pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 3. Hugging Face 生態系 (相容性鎖定)
print("   ⬇️ 安裝 Hugging Face Stack...")
# Hub 0.27+ 解決 DryRunError
# Transformers 4.48+ 解決 Gemma 2 bug
!pip install -U "huggingface-hub>=0.27.0"
!pip install -U "transformers>=4.48.0"
!pip install -U accelerate bitsandbytes peft datasets

# 4. RAG 與應用層
print("   ⬇️ 安裝應用層依賴...")
!pip install -U sentence-transformers faiss-cpu pydub
!pip install -U pillow==11.0.0 librosa soundfile
!pip install -U qrcode[pil] albumentations==1.3.1 opencv-python-headless gTTS edge-tts nest_asyncio pyttsx3

# 5. Gradio (確保最新)
!pip install -U gradio>=4.0.0

print("   ✅ 所有依賴安裝完成！")

# %%
# ============================================================================
# STEP 5: 自我檢測與啟動
# ============================================================================
print("\n[6/6] 系統自我檢測...")

try:
    import torch
    import torchvision
    import transformers
    import huggingface_hub
    
    print(f"   🔍 Torch Version: {torch.__version__}")
    print(f"   🔍 Vision Version: {torchvision.__version__}")
    print(f"   🔍 Transformers: {transformers.__version__}")
    print(f"   🔍 Hub Version: {huggingface_hub.__version__}")
    
    # 簡單的 GPU 檢查
    if torch.cuda.is_available():
        print(f"   🔍 GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        print("   ⚠️ WARNING: No GPU detected! Inference will be slow.")

except ImportError as e:
    print(f"   ❌ CRITICAL: 環境檢測失敗 - {e}")
    # 這裡不拋出錯誤，嘗試繼續執行

from huggingface_hub import login
login(token=hf_token)

print("\n" + "=" * 80)
print("🚀 啟動 SilverGuard: Impact Research Edition (V10.0 Final)")
print("=" * 80)

%run SilverGuard_Impact_Research_V8.py
