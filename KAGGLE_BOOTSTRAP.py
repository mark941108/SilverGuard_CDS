"""
================================================================================
🏥 AI Pharmacist Guardian - Kaggle Bootstrap Script
================================================================================
📋 使用方式：
   1. 在 Kaggle Notebook 中新建一個 Cell
   2. 複製貼上此腳本並執行
   3. 腳本會自動下載代碼、安裝依賴、執行訓練

⚠️ 前置要求：
   - 在 Add-ons > Secrets 中設定 GITHUB_TOKEN
   - 在 Add-ons > Secrets 中設定 HUGGINGFACE_TOKEN
   - 已接受 MedGemma License (https://huggingface.co/google/medgemma-1.5-4b-it)
================================================================================
"""

# %%
# ============================================================================
# STEP 0: 認證與環境設置
# ============================================================================
from kaggle_secrets import UserSecretsClient
import os

print("=" * 80)
print("🏥 AI Pharmacist Guardian - Kaggle Bootstrap")
print("=" * 80)

# 讀取 Secrets
user_secrets = UserSecretsClient()

print("\n[1/4] 讀取認證金鑰...")
try:
    gh_token = user_secrets.get_secret("GITHUB_TOKEN")
    print("   ✅ GITHUB_TOKEN 已讀取")
except:
    print("   ❌ GITHUB_TOKEN 未設定！請去 Add-ons > Secrets 設定")
    gh_token = ""

try:
    hf_token = user_secrets.get_secret("HUGGINGFACE_TOKEN")
    print("   ✅ HUGGINGFACE_TOKEN 已讀取")
except:
    print("   ❌ HUGGINGFACE_TOKEN 未設定！請去 Add-ons > Secrets 設定")
    hf_token = ""

# %%
# ============================================================================
# STEP 1: 下載 Repository
# ============================================================================
print("\n[2/4] 下載 SilverGuard Repository...")

# 清理舊環境
!rm -rf SilverGuard
!rm -rf medgemma_training_data_v5

# Clone Repository
repo_url = f"https://{gh_token}@github.com/mark941108/SilverGuard.git"
!git clone --depth 1 {repo_url}

# 進入工作目錄
%cd SilverGuard

print("   ✅ Repository 下載完成")

# %%
# ============================================================================
# STEP 2: 安裝依賴
# ============================================================================
print("\n[3/4] 安裝依賴套件...")

# 核心依賴
!pip install -q -U "huggingface-hub<1.0" bitsandbytes peft accelerate datasets
!pip install -q transformers>=4.50.0
!pip install -q pillow==11.0.0 torchaudio librosa soundfile

# 專案依賴
!pip install -q qrcode[pil] albumentations==1.3.1 opencv-python-headless gTTS edge-tts

print("   ✅ 依賴安裝完成")

# %%
# ============================================================================
# STEP 3: HuggingFace 登入
# ============================================================================
print("\n[4/4] HuggingFace 登入...")

from huggingface_hub import login
login(token=hf_token)
print("   ✅ HuggingFace 登入成功")

# %%
# ============================================================================
# STEP 4: 執行主程式
# ============================================================================
print("\n" + "=" * 80)
print("\n" + "=" * 80)
print("🚀 啟動 SilverGuard: Impact Research Edition (V8.2)")
print("=" * 80)

# 🔥 正確的檔名 (Updated for V8)
%run SilverGuard_Impact_Research_V8.py

print("\n" + "=" * 80)
print("🎉 執行完成！請查看上方輸出")
print("=" * 80)
