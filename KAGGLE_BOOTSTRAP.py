"""
================================================================================
🏥 AI Pharmacist Guardian - Kaggle Bootstrap (V12.8 Omni-Nexus)
================================================================================
📋 戰略更新對應 (V12.8 Final Fix):
   1. [SMART SYNC] 優先使用本地上傳檔案 (Local Override Mode)。
      解決：修正了「本地修改後，Bootstrap 卻從 GitHub 拉取舊版」的邏輯死循環。
      如果偵測到 SilverGuard_Impact_Research_V8.py 存在，直接使用，不 Clone。
   2. [STABILITY] 鎖定 2026 T4 黃金組合 (PyTorch 2.6.0 + cu118)。
      解決：避免使用不穩定的 cu12x，改用最成熟的 cu118。
   3. [SAFETY] 鎖定 Transformers 4.48+ (避開 5.0.0 早期風險)。
================================================================================
"""

# %%
# ============================================================================
# STEP 0: 環境重置與認證
# ============================================================================
import os
import sys
import shutil # Added for Smart Sync
import re
from kaggle_secrets import UserSecretsClient

print("=" * 80)
print("🏥 AI Pharmacist Guardian - Bootstrap (V12.8 Omni-Nexus)")
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
# 使用 Regex 強制關閉，解決 PyTorch 2.6 的潛在 Deadlock 問題
if re.search(r"gradient_checkpointing\s*=\s*True", content):
    print("   🔧 手術 B: 強制關閉 Gradient Checkpointing (Fix Deadlock)...")
    content = re.sub(r"gradient_checkpointing\s*=\s*True", "gradient_checkpointing=False", content)

# --- 手術 C: 防止 OOM (Reduce Batch Size) ---
# --- 手術 C: 防止 OOM (Reduce Batch Size) ---
content = re.sub(r"per_device_train_batch_size\s*=\s*\d+", "per_device_train_batch_size=1", content)
content = re.sub(r"gradient_accumulation_steps\s*=\s*\d+", "gradient_accumulation_steps=4", content)

# --- 手術 E: 硬體加速 (CuDNN Benchmark) ---
if "torch.backends.cudnn.benchmark" not in content:
    print("   🔧 手術 E: 啟用 CuDNN Benchmark (Hardware Optimization)...")
    # 在 import torch 之後插入（假設文件開頭有 import，或者我們插入在開頭附近）
    # 更安全的方法是找個穩定的插入點，例如在 STEP 0 或 STEP 1 的 log 之後，或者直接在開頭 import block 後
    # 這裡我們選擇直接在 main block 開始處插入，或替換一個已知的行
    # 簡單暴力：在 content 開頭加入
    content = "import torch\ntorch.backends.cudnn.benchmark = True\n" + content

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

# 4. Hugging Face Stack (鎖定穩定版)
# 建議使用 4.48+ 以支援 Gemma 3 架構，避開剛發布的 5.0.0 潛在 bug
print("   ⬇️ 安裝 Hugging Face Stack...")
!pip install -U "huggingface-hub>=0.27.0" "transformers>=4.48.0,<5.0.0" accelerate bitsandbytes peft datasets

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
print("🚀 啟動 SilverGuard: Impact Research Edition (V12.8 Omni-Nexus)")
print("=" * 80)

# 執行
%run SilverGuard_Impact_Research_V8.py
