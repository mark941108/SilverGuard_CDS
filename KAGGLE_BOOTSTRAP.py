"""
================================================================================
🏥 AI Pharmacist Guardian - Kaggle Bootstrap (V12.0 Final Anti-Deadlock)
================================================================================
📋 戰略總監的最終解決方案：
   1. [PLATINUM] 使用 PyTorch 2.6.0 + Transformers 5.0.0 (解決依賴報錯)。
   2. [SURGERY]  自動修改代碼，強制關閉 Gradient Checkpointing (解決 T4 死鎖)。
   3. [HOTFIX]   自動修復遺失的 'QD_meals_with' 鍵值 (解決 KeyError)。
================================================================================
"""

# %%
# ============================================================================
# STEP 0: 環境重置與認證
# ============================================================================
import os
import sys
import re
from kaggle_secrets import UserSecretsClient

print("=" * 80)
print("🏥 AI Pharmacist Guardian - Bootstrap (V12.0 Anti-Deadlock)")
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
# STEP 2: 執行「開腦手術」 (Critical Surgery) - 修復所有已知問題
# ============================================================================
print("\n[3/6] 正在對代碼進行外科手術 (修復死鎖與錯誤)...")

target_file = "SilverGuard_Impact_Research_V8.py"

with open(target_file, "r", encoding="utf-8") as f:
    content = f.read()

# --- 手術 A: 修復 KeyError (QD_meals_with) ---
if '"QD_meals_with":' not in content:
    print("   🔧 手術 A: 注入遺失的藥物代碼 (Fix KeyError)...")
    patch_code = """
    "QD_breakfast_after": {"code": "QD-PC", "zh": "每日1次，早餐後服用", "detail": "每日早餐後30分鐘服用"},
    "QD_meals_with": {"code": "QD-M", "zh": "每日1次，隨餐服用", "detail": "請於用餐時一併服用以增加吸收"},
    """
    anchor = '"QD_breakfast_after": {"code": "QD-PC", "zh": "每日1次，早餐後服用", "detail": "每日早餐後30分鐘服用"},'
    content = content.replace(anchor, patch_code.strip())

# --- 手術 B: 解除死鎖 (Disable Gradient Checkpointing) ---
# 這是導致您卡在 "30分鐘沒動" 的元兇
if "gradient_checkpointing=True" in content:
    print("   🔧 手術 B: 強制關閉 Gradient Checkpointing (Fix Deadlock)...")
    content = content.replace("gradient_checkpointing=True", "gradient_checkpointing=False")

# --- 手術 C: 防止 OOM (Reduce Batch Size) ---
# 因為關閉了 Checkpointing，VRAM 會吃緊，必須把 Batch Size 降到 1
print("   🔧 手術 C: 調整 Batch Size 為 1 以防記憶體溢出...")
content = re.sub(r"per_device_train_batch_size\s*=\s*\d+", "per_device_train_batch_size=1", content)
content = re.sub(r"gradient_accumulation_steps\s*=\s*\d+", "gradient_accumulation_steps=8", content)

# 寫回檔案
with open(target_file, "w", encoding="utf-8") as f:
    f.write(content)

print("   ✅ 手術完成！代碼已準備好在 T4 上穩定運行。")

# %%
# ============================================================================
# STEP 3: 暴力清除舊環境 (The Nuke)
# ============================================================================
print("\n[4/6] 清理衝突套件...")
# 為了 Save and Run All 的穩定性，我們不假設環境是乾淨的
!pip uninstall -y torch torchvision torchaudio transformers huggingface_hub sentence-transformers accelerate peft bitsandbytes gradio

# %%
# ============================================================================
# STEP 4: 乾淨安裝 (The Pave) - V11.0 白金依賴矩陣
# ============================================================================
print("\n[5/6] 安裝白金版本組合 (PyTorch 2.6 + Transformers 5.0)...")

# 1. 系統依賴
!apt-get update -y && apt-get install -y libespeak1 libsndfile1 ffmpeg

# 2. PyTorch 2.6.0 (解決 ValueError)
print("   ⬇️ 安裝 PyTorch 2.6.0 Ecosystem...")
!pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 3. Hugging Face Stack (Latest)
print("   ⬇️ 安裝 Hugging Face Stack...")
!pip install -U "huggingface-hub>=0.27.0"
!pip install -U "transformers>=5.0.0"
!pip install -U accelerate bitsandbytes peft datasets

# 4. RAG 與應用層
print("   ⬇️ 安裝應用層依賴...")
!pip install -U sentence-transformers faiss-cpu pydub
!pip install -U pillow==11.0.0 librosa soundfile
!pip install -U qrcode[pil] albumentations==1.3.1 opencv-python-headless gTTS edge-tts nest_asyncio pyttsx3

# 5. Gradio
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
print("🚀 啟動 SilverGuard: Impact Research Edition (V12.0 Final)")
print("=" * 80)

# 執行
%run SilverGuard_Impact_Research_V8.py
