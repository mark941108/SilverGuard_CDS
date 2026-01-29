"""
================================================================================
🏥 AI Pharmacist Guardian - Kaggle Bootstrap Script (V9.5 Diamond Lock)
================================================================================
📋 戰略總監認證：
   1. [LOCK] 嚴格鎖定 Transformer=4.46.1, Hub=0.26.2 (拒絕未來版本的不穩定性)。
   2. [CLEAN] 啟動前強制清除記憶體中的衝突模組。
   3. [STABLE] 使用 PyTorch 2.5.1 黃金標準。
================================================================================
"""

# %%
# ============================================================================
# STEP 0: 記憶體除魔 (Magic Wipe) & 認證
# ============================================================================
import sys
import os

# 強制從記憶體中移除可能衝突的庫 (防止 Zombie Kernel)
modules_to_kill = ["transformers", "huggingface_hub", "torch", "torchvision"]
for m in modules_to_kill:
    if m in sys.modules:
        print(f"🧹 清除記憶體殘留: {m}")
        del sys.modules[m]

from kaggle_secrets import UserSecretsClient

print("=" * 80)
print("🏥 AI Pharmacist Guardian - Bootstrap (V9.5 Diamond Lock)")
print("=" * 80)

user_secrets = UserSecretsClient()
print("\n[1/5] 讀取認證金鑰...")
try:
    gh_token = user_secrets.get_secret("GITHUB_TOKEN")
    hf_token = user_secrets.get_secret("HUGGINGFACE_TOKEN")
    print("   ✅ 金鑰讀取成功")
except:
    print("   ❌ 金鑰未設定！")
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
print("\n[3/5] 應用代碼熱修復...")
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
# STEP 3: 安裝依賴 (鑽石級鎖定版)
# ============================================================================
print("\n[4/5] 安裝依賴套件 (Diamond Configuration)...")

# 1. [SYSTEM] 系統庫
!apt-get update -y && apt-get install -y libespeak1 libsndfile1 ffmpeg

# 2. [CORE FIX] 重置 PyTorch 到 2.5.1
print("   🔧 重置 PyTorch (2.5.1 Stable)...")
!pip uninstall -y torch torchvision torchaudio transformers huggingface_hub
!pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 3. [ML] 鎖定 Transformers 與 Hub (這兩個版本是 100% 兼容的)
# huggingface_hub 0.26.2 包含 DryRunError
# transformers 4.46.1 完美支援 Gemma 2 且不會抓狂
print("   🔧 安裝 ML 核心庫 (Locked Versions)...")
!pip install -q -U "huggingface-hub==0.26.2"
!pip install -q -U "transformers==4.46.1" accelerate bitsandbytes peft datasets

# 4. [RAG & APP] 應用層依賴
print("   🔧 安裝 RAG 與應用工具...")
!pip install -q -U sentence-transformers==3.2.1 faiss-cpu pydub
!pip install -q pillow==11.0.0 librosa soundfile
!pip install -q qrcode[pil] albumentations==1.3.1 opencv-python-headless gTTS edge-tts nest_asyncio pyttsx3

print("   ✅ 鑽石級依賴安裝完成！")

# %%
# ============================================================================
# STEP 4: 執行主程式
# ============================================================================
print("\n[5/5] 啟動主程式...")
import huggingface_hub
from huggingface_hub import login
print(f"   🔍 Debug: Hub Version = {huggingface_hub.__version__}") # 應該顯示 0.26.2

login(token=hf_token)

print("\n" + "=" * 80)
print("🚀 啟動 SilverGuard: Impact Research Edition (V8.2 + Diamond)")
print("=" * 80)

%run SilverGuard_Impact_Research_V8.py
