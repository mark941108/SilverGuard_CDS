"""
================================================================================
🏥 AI Pharmacist Guardian - Kaggle Bootstrap (V12.13 Gemma 3 Fix)
================================================================================
📋 戰略更新對應 (V12.13 Hotfix):
   1. [UPGRADE] 升級 Transformers 至 >= 4.51.0 (支援 Gemma 3)。
      原因：MedGemma 1.5 使用 Gemma 3 架構，舊版 4.47.1 發生 Model Type Error。
      風險管理：DryRunError 預期已由 V8.py 的 pip 禁用 (Silence Internal Pip) 解決。
   2. [CLEANUP] 保持移除「手術刀邏輯」。
================================================================================
"""

# %%
# ============================================================================
# STEP 0: 環境重置與認證
# ============================================================================
import os
import sys
import shutil 
import re
from kaggle_secrets import UserSecretsClient

print("=" * 80)
print("🏥 AI Pharmacist Guardian - Bootstrap (V12.13 Gemma 3 Fix)")
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
# STEP 1: 智慧型部署 (Smart Sync) - 2026 V12.8 Edition
# ============================================================================
print("\n[2/6] 部署 SilverGuard (優先權: 本地上傳 > GitHub Clone)...")

# 1. 定義關鍵檔案 (用於偵測是否為手動上傳模式)
target_file = "SilverGuard_Impact_Research_V8.py"

# 檢查 Kaggle 根目錄是否有你剛剛上傳/修改的檔案
if os.path.exists(target_file):
    # 【場景 A】你手動上傳了修復檔 -> 使用本地檔，不准 Git 覆蓋
    print(f"   ✅ 偵測到本地檔案：{target_file}")
    print("   🚀 啟動 [Local Override Mode]：略過 GitHub Clone，使用當前版本。")
    
    # 建立目錄結構 (模擬 Clone 後的資料夾結構，以免後續 %cd 失敗)
    os.makedirs("SilverGuard", exist_ok=True)
    
    # 將根目錄的所有 .py 檔案複製進去 (保留你的修改)
    # Note: !cp in python script context might need os.system or shutil, 
    # but in Jupyter !cp works. Since this is a .py file intended for Jupyter, we keep ! syntax if compatible
    # or use shutil for pure python safety. Let's use shutil for robustness in python script.
    # Actually, the user provided code uses !cp, so we stick to it for Jupyter compatibility.
    !cp *.py SilverGuard/ 2>/dev/null
    
else:
    # 【場景 B】乾淨環境 -> 從 GitHub 拉取
    print("   ☁️ 未偵測到本地檔案，啟動 [GitHub Clone Mode]...")
    !rm -rf SilverGuard
    
    # [FIX] 防止 Git Auth 卡死 (The Silent Hang Fix)
    # 只有在真的有 token 時才加入 @，否則 Git 會跳出隱形密碼輸入框導致卡死
    if gh_token:
        repo_url = f"https://{gh_token}@github.com/mark941108/SilverGuard.git"
    else:
        print("   ⚠️ 無 GitHub Token，嘗試 Public Clone (無密碼模式)...")
        repo_url = "https://github.com/mark941108/SilverGuard.git"
        
    !git clone --depth 1 {repo_url}
    print("   ✅ Repository 下載完成")

# 進入目錄
%cd SilverGuard
print(f"   📂 當前工作目錄: {os.getcwd()}")

# %%
# ============================================================================
# STEP 2: (SKIPPED) 移除手術刀邏輯 - 直接使用乾淨代碼
# ============================================================================
print("\n[3/6] Skipping Surgery (Using Clean Code V8)...")
# 原本這裡有 Regex Replace 代碼，現已移除以確保穩定性。
# 請確保上傳的 SilverGuard_Impact_Research_V8.py 已經包含正確的 eGFR 邏輯。

# %%
# ============================================================================
# STEP 3: 暴力清除舊環境 (The Nuke)
# ============================================================================
print("\n[4/6] 清理衝突套件 (Aggressive Torch Removal)...")
# V12.7: 強制移除 torch 相關套件，避免 pip 認為 "Requirement satisfied" 而跳過升級
!pip uninstall -y torch torchvision torchaudio transformers huggingface_hub sentence-transformers accelerate peft bitsandbytes gradio

# %%
# ============================================================================
# STEP 4: 乾淨安裝 (The Pave) - V12.8 白金依賴矩陣
# ============================================================================
print("\n[5/6] 安裝白金版本組合 (PyTorch 2.6.0 + cu118)...")

# 1. 系統依賴 (TTS & Audio 必備)
!apt-get update -y && apt-get install -y libespeak1 libsndfile1 ffmpeg

# 2. 暴力移除舊版 (防止 Version Conflict)
print("   ☢️ 清理衝突套件...")
!pip uninstall -y torch torchvision torchaudio transformers huggingface_hub

# 3. PyTorch 2.6.0 (Stable for T4 in 2026)
# 指定 cu118 版本以獲得最佳穩定性，避免 cu121/cu124 相容性問題
print("   ⬇️ 安裝 PyTorch 2.6.0 Ecosystem (CUDA 11.8)...")
!pip install --no-cache-dir torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# 4. Hugging Face Stack (升級支援 Gemma 3)
# 原因: Gemma 3 架構需要最新版 Transformers (>=4.51.0)
# 修正: 不再鎖定 4.47.1，改為安裝最新穩定版
print("   ⬇️ 安裝 Hugging Face Stack (Gemma 3 Support)...")
!pip install -U "huggingface-hub>=0.29.0" "transformers>=4.51.0" accelerate bitsandbytes peft datasets

# 5. 應用層依賴 (RAG, Vision, Audio)
print("   ⬇️ 安裝應用層依賴...")
!pip install -U sentence-transformers faiss-cpu pydub
!pip install -U pillow==11.0.0 librosa soundfile
!pip install -U qrcode[pil] albumentations==1.3.1 opencv-python-headless gTTS edge-tts nest_asyncio pyttsx3
!pip install -U gradio>=4.0.0

print("   ✅ 所有依賴安裝完成！")

# %%
# ============================================================================
# STEP 5: 啟動主程式
# ============================================================================
print("\n[6/6] 系統啟動...")

from huggingface_hub import login
login(token=hf_token)

print("\n" + "=" * 80)
print("🚀 啟動 SilverGuard: Impact Research Edition (V12.13 Gemma 3 Fix)")
print("=" * 80)

# 執行
%run SilverGuard_Impact_Research_V8.py
