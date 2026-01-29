"""
================================================================================
🏥 AI Pharmacist Guardian - Kaggle Bootstrap (V12.2 Platinum Stable)
================================================================================
📋 戰略更新對應 (V12.2):
   1. [CRITICAL] 強制鎖定 Transformers < 5.0.0。
      原因：Transformers 5.0.0 引入了 Gemma 3 架構，強制要求 PyTorch >= 2.6.0。
      為了維持 T4 穩定性 (使用 PyTorch 2.5.1)，必須禁止升級到 5.0。
   2. [LOGIC] 維持 Metformin eGFR 檢查邏輯。
   3. [COMPLIANCE] 維持藥師法第 19 條標示。
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
print("🏥 AI Pharmacist Guardian - Bootstrap (V12.2 Platinum Stable)")
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
print("\n[3/6] 正在對代碼進行外科手術 (V12.2 Logic Updates)...")

target_file = "SilverGuard_Impact_Research_V8.py"

with open(target_file, "r", encoding="utf-8") as f:
    content = f.read()

# --- 手術 A: 修正 Metformin 邏輯 (Hard Rule -> Missing Data) ---
# 將 Metformin > 1000mg 的硬性 HIGH_RISK 警告改為 MISSING_DATA (如果尚未修改)
if 'safety["status"] = "HIGH_RISK"' in content and 'Metformin > 1000mg' in content:
    print("   🔧 手術 A: 修正 Metformin 規則 (High Risk -> Missing Data)...")
    content = content.replace(
        'safety["status"] = "HIGH_RISK"',
        'safety["status"] = "MISSING_DATA"'
    ).replace(
        'safety["reasoning"] = "⚠️ [System Hard Rule] Metformin 每日劑量超過 1000mg，對於腎功能衰退的老年人具有高度乳酸中毒風險。"',
        'safety["reasoning"] = "⚠️ [AGS Beers Criteria] 偵測到 Metformin 高劑量，但缺少腎功能數據(eGFR)。請確認 eGFR > 30 mL/min 以確保安全。"'
    )

# --- 手術 B: 解除死鎖 (Disable Gradient Checkpointing) ---
if "gradient_checkpointing=True" in content:
    print("   🔧 手術 B: 強制關閉 Gradient Checkpointing (Fix Deadlock)...")
    content = content.replace("gradient_checkpointing=True", "gradient_checkpointing=False")

# --- 手術 C: 防止 OOM (Reduce Batch Size) ---
content = re.sub(r"per_device_train_batch_size\s*=\s*\d+", "per_device_train_batch_size=1", content)
content = re.sub(r"gradient_accumulation_steps\s*=\s*\d+", "gradient_accumulation_steps=8", content)

# --- 手術 D: 修復縮排錯誤 (Extra Safety) ---
# 針對 User 之前回報的 IndentationError 進行防禦性檢查
# 雖然 User 說已經修復，但 Bootstrap 手術可能會再次觸發
# 這裡我們不做 Blind Regex Replace，相信 Git Pull 下來的版本已經修復

# 寫回檔案
with open(target_file, "w", encoding="utf-8") as f:
    f.write(content)

print("   ✅ V12.2 手術完成！")

# %%
# ============================================================================
# STEP 3: 暴力清除舊環境 (The Nuke)
# ============================================================================
print("\n[4/6] 清理衝突套件...")
!pip uninstall -y torch torchvision torchaudio transformers huggingface_hub sentence-transformers accelerate peft bitsandbytes gradio

# %%
# ============================================================================
# STEP 4: 乾淨安裝 (The Pave) - V12.2 白金依賴矩陣
# ============================================================================
print("\n[5/6] 安裝白金版本組合 (PyTorch 2.5.1 + Transformers 4.x)...")

# 1. 系統依賴
!apt-get update -y && apt-get install -y libespeak1 libsndfile1 ffmpeg

# 2. PyTorch 2.5.1 (Stable Golden Config)
print("   ⬇️ 安裝 PyTorch 2.5.1 Ecosystem...")
!pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# 3. Hugging Face Stack (PINNED VERSION)
# 🔥 V12.2 CRITICAL FIX: 禁止安裝 Transformers 5.0+
print("   ⬇️ 安裝 Hugging Face Stack (Forced Transformers 4.x)...")
!pip install -U "huggingface-hub>=0.26.0"
!pip install -U "transformers>=4.46.0,<5.0.0"
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
print("🚀 啟動 SilverGuard: Impact Research Edition (V12.2 Platinum)")
print("=" * 80)

# 執行
%run SilverGuard_Impact_Research_V8.py
