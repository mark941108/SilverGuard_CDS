"""
================================================================================
🏥 AI Pharmacist Guardian - Kaggle Bootstrap Script (V9.4 Golden Config)
================================================================================
📋 戰略總監認證：
   1. [CORE] 強制降級 PyTorch 至 2.5.1 (Stable)，解決 torchvision 崩潰。
   2. [FIX] 解鎖 huggingface_hub 版本，解決 DryRunError。
   3. [RAG] 確保 sentence-transformers 與 faiss-cpu 正確安裝。
================================================================================
"""

# %%
# ============================================================================
# STEP 0: 認證與環境設置
# ============================================================================
from kaggle_secrets import UserSecretsClient
import os

print("=" * 80)
print("🏥 AI Pharmacist Guardian - Kaggle Bootstrap (V9.4 Golden Config)")
print("=" * 80)

user_secrets = UserSecretsClient()
print("\n[1/5] 讀取認證金鑰...")
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
print("\n[2/5] 下載 SilverGuard Repository...")
!rm -rf SilverGuard medgemma_training_data_v5
repo_url = f"https://{gh_token}@github.com/mark941108/SilverGuard.git"
!git clone --depth 1 {repo_url}
%cd SilverGuard
print("   ✅ Repository 下載完成")

# %%
# ============================================================================
# STEP 2: 自動熱修復 (Hotfix Patch)
# ============================================================================
print("\n[3/5] 應用代碼熱修復 (Hotfix)...")
# 注入遺失的藥物用法鍵值
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
            print("   ✅ 熱修復成功 (QD_meals_with 注入)")
    else:
        print("   ✅ 代碼已包含修復")
except Exception as e:
    print(f"   ⚠️ 熱修復跳過: {e}")

# %%
# ============================================================================
# STEP 3: 安裝依賴 (黃金組合版)
# ============================================================================
print("\n[4/5] 安裝依賴套件 (Golden Configuration)...")

# 1. [SYSTEM] 系統庫 (TTS/Audio)
!apt-get update -y && apt-get install -y libespeak1 libsndfile1 ffmpeg

# 2. [CORE FIX] 重置 PyTorch 到工業穩定版 (2.5.1)
# 這是解決 'partially initialized module' 的唯一方法
print("   🔧 正在重置 PyTorch 環境 (這可能需要 1-2 分鐘)...")
!pip uninstall -y torch torchvision torchaudio
!pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 3. [ML] 安裝支援 Gemma 2 的 Transformers 與 Hub
# 解鎖 huggingface_hub 以修復 DryRunError
print("   🔧 安裝 ML 核心庫...")
!pip install -q -U "huggingface-hub>=0.26.0" 
!pip install -q -U "transformers>=4.46.0" accelerate bitsandbytes peft datasets

# 4. [RAG & APP] 應用層依賴
print("   🔧 安裝 RAG 與應用工具...")
!pip install -q -U sentence-transformers==3.3.1 faiss-cpu pydub
!pip install -q pillow==11.0.0 librosa soundfile
!pip install -q qrcode[pil] albumentations==1.3.1 opencv-python-headless gTTS edge-tts nest_asyncio pyttsx3

print("   ✅ 黃金組合安裝完成！")

# %%
# ============================================================================
# STEP 4: 執行主程式
# ============================================================================
print("\n[5/5] 啟動主程式...")
from huggingface_hub import login
login(token=hf_token)

print("\n" + "=" * 80)
print("🚀 啟動 SilverGuard: Impact Research Edition (V8.2 + Golden)")
print("=" * 80)

%run SilverGuard_Impact_Research_V8.py
