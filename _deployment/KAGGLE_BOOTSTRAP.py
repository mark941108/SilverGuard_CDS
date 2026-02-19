"""
================================================================================
🏥 AI Pharmacist Guardian - Kaggle Bootstrap (V12.16 Impact)
================================================================================
📋 戰略更新對應 (V12.13 Hotfix):
   1. [UPGRADE] 升級 Transformers 至 >= 4.51.0 (支援 Gemma 3)。
      原因：MedGemma 1.5 使用 Gemma 3 架構，確保 SigLIP 編碼器兼容性。
      風險管理：DryRunError 預期已由 V8.py 的 pip 禁用 (Silence Internal Pip) 解決。
   2. [CLEANUP] 保持移除「手術刀邏輯」。
================================================================================
"""

# %%
# ============================================================================
# 📦 DATASET LOADER (Auto-Copy from /kaggle/input) - [V12.16 Impact]
# ============================================================================
# This script is designed to run in Kaggle Kernels. It scans /kaggle/input for
# critical files (agent_engine.py, medgemma_data.py, fonts) and copies them
# to the working directory. This enables "Local Override Mode" without Git.

import os
import sys
import shutil
import re
from kaggle_secrets import UserSecretsClient
import subprocess

print("🔍 Scanning for SilverGuard assets in /kaggle/input...")
target_files = [
    "agent_engine.py", 
    "agent_utils.py",
    "medgemma_data.py", 
    "app.py", 
    "tts_engine.py",
    "piper_engine.py",
    "requirements.txt",
    "generate_v17_fusion.py", 
    "generate_stress_test.py",
    "viewer.min.css",
    "viewer.min.js",
    "Writeup.md",
    "NotoSansTC-Bold.otf",
    "NotoSansTC-Regular.otf"
]
files_copied = 0
for root, dirs, files in os.walk("/kaggle/input"):
    for file in files:
        if file in target_files:
            src = os.path.join(root, file)
            dst_folder = os.getcwd()
            if file.endswith(".otf") or file.endswith(".ttf"):
                dst_folder = os.path.join(os.getcwd(), "assets", "fonts")
            elif file.endswith(".md"):
                dst_folder = os.path.join(os.getcwd(), "_documentation")
            # [KAGGLE FIX] Copy demo images to assets/DEMO
            elif file.endswith(".png") or file.endswith(".jpg"):
                dst_folder = os.path.join(os.getcwd(), "assets", "DEMO")
            
            os.makedirs(dst_folder, exist_ok=True)
            dst = os.path.join(dst_folder, file)
            if not os.path.exists(dst):
                try:
                    shutil.copy2(src, dst)
                    print(f"   📂 Loaded: {file} -> {dst_folder}")
                    files_copied += 1
                except Exception as e:
                    print(f"   ⚠️ Failed to copy {file}: {e}")
if files_copied > 0:
    print(f"✅ Successfully loaded {files_copied} assets from Dataset.")
else:
    print("ℹ️ No external dataset assets found. Assuming GitHub Clone mode or Local run.")

# ============================================================================
# STEP 0: 環境重置與認證
# ============================================================================


print("=" * 80)
print("🏥 AI Pharmacist Guardian - Bootstrap (V12.16 Impact)")
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
# ✅ [Omni-Nexus Fix] 檢查所有必要檔案 (防止漏傳 medgemma_data.py 導致崩潰)
target_file = "agent_engine.py"
required_files = ["agent_engine.py", "medgemma_data.py"]
missing_files = [f for f in required_files if not os.path.exists(f)]

# 檢查 Kaggle 根目錄是否有完整檔案
if not missing_files:
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
    # [Fix] Use os.system for compatibility
    import subprocess
    try:
        # [FIX] Also copy font files (.otf) so the generator can find them
        subprocess.run("cp *.py *.otf SilverGuard/", shell=True, check=True, stderr=subprocess.DEVNULL)
        print("   ✅ Synced Python scripts & Fonts to SilverGuard sandbox")
    except:
        pass
    
else:
    # 【場景 B】乾淨環境 -> 從 GitHub 拉取
    print("   ☁️ 未偵測到本地檔案，啟動 [GitHub Clone Mode]...")
    if os.path.exists("SilverGuard"):
        shutil.rmtree("SilverGuard")
    
    # [FIX] 防止 Git Auth 卡死 (The Silent Hang Fix)
    # 只有在真的有 token 時才加入 @，否則 Git 會跳出隱形密碼輸入框導致卡死
    if gh_token:
        repo_url = f"https://{gh_token}@github.com/mark941108/SilverGuard.git"
    else:
        repo_url = "https://github.com/mark941108/SilverGuard.git"
    subprocess.run(f"git clone --depth 1 {repo_url}", shell=True, check=True)
    print("   ✅ Repository 下載完成")

# 進入目錄
# ✅ [Omni-Nexus Fix] 防止重複進入子目錄導致的路徑混亂
if os.path.basename(os.getcwd()) != "SilverGuard":
    if os.path.exists("SilverGuard"):
        os.chdir("SilverGuard")
        print(f"   📂 已進入目錄: {os.getcwd()}")
    else:
        print("❌ 錯誤：找不到 SilverGuard 目錄")

print("\n[2.5/6] 正在校準系統參數 (Threshold & GPU Stability)...")
try:
    # 1. 原本的模糊閾值校準
    # 1. 原本的模糊閾值校準 (User requested 25.0, which is default)
    # subprocess.run(["sed", "-i", "s/BLUR_THRESHOLD = 25.0/BLUR_THRESHOLD = 10.0/g", "medgemma_data.py"], check=True)
    
    # 2. 🟢 【熱修復 V9.2】暴力鎖定 T4 算力防護與處理器模式
    # 強制將所有文件中的 cudnn.benchmark 設為 False (解決 T4 VRAM 碎裂)
    subprocess.run("sed -i 's/torch.backends.cudnn.benchmark = .*/torch.backends.cudnn.benchmark = False/g' *.py", shell=True, check=True)
    subprocess.run("sed -i 's/print(\".* CuDNN Benchmark .*\")/print(\"🛡️ CuDNN Benchmark Disabled (Global Stability Mode)\")/g' *.py", shell=True, check=True)
    
    # 強制鎖定 Gemma 3 慢速處理器模式 (解決 T4 float16 斷氣問題)
    # 此指令會掃描所有 AutoProcessor.from_pretrained 調用並注入 use_fast=False
    subprocess.run("sed -i 's/AutoProcessor.from_pretrained(\\\\([^)]*\\\\))/AutoProcessor.from_pretrained(\\\\1, use_fast=False)/g' *.py", shell=True, check=True)
    # [Fix] 避免重複注入 use_fast=False
    subprocess.run("sed -i 's/use_fast=False, use_fast=False/use_fast=False/g' *.py", shell=True, check=True)
    
    # 修正 Gemma 3Processor 警告日誌
    print("   ✅ 環境參數與 GPU 防護校準完成 (Hot-Patch V8.7 Aggressive Enforcement)")
except Exception as e:
    print(f"   ⚠️ 校準失敗: {e}")
    pass

os.environ["OFFLINE_MODE"] = "True"
print("   🔒 OFFLINE_MODE = True")

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
import subprocess
try:
    subprocess.run("pip uninstall -y torch torchvision torchaudio transformers huggingface_hub sentence-transformers accelerate peft bitsandbytes gradio", shell=True, check=True)
except:
    pass

# %%
# ============================================================================
# STEP 4: 乾淨安裝 (The Pave) - V12.8 白金依賴矩陣
# ============================================================================
print("\n[5/6] 安裝白金版本組合 (PyTorch 2.6.0 + cu118)...")

# 1. 系統依賴 (TTS & Audio 必備)
# 1. 系統依賴 (TTS & Audio 必備 + 中文字型)
subprocess.run("apt-get update -y && apt-get install -y libespeak1 libsndfile1 ffmpeg fonts-noto-cjk", shell=True, check=True)

# 2. 暴力移除舊版 (防止 Version Conflict)
print("   ☢️ 清理衝突套件...")
try:
    subprocess.run("pip uninstall -y torch torchvision torchaudio transformers huggingface_hub opencv-python", shell=True, check=True)
except:
    pass

# 3. PyTorch 2.6.0 (Stable for T4 in 2026)
# 指定 cu118 版本以獲得最佳穩定性，避免 cu121/cu124 相容性問題
print("   ⬇️ 安裝 PyTorch 2.6.0 Ecosystem (CUDA 11.8)...")
subprocess.run("pip install --no-cache-dir torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 --index-url https://download.pytorch.org/whl/cu118", shell=True, check=True)
subprocess.run('pip install -U "transformers>=4.51.0" "accelerate>=1.3.0" "bitsandbytes>=0.45.0" "peft>=0.14.0"', shell=True, check=True)
subprocess.run('pip install -U "gradio>=5.15.0" "fastapi>=0.115.0,<0.124.0" "pydantic>=2.10.0"', shell=True, check=True)
subprocess.run('pip uninstall -y pillow matplotlib', shell=True) 
# [極度重要] 確保安裝了 edge-tts
subprocess.run('pip install -U "pillow>=10.4.0" "matplotlib>=3.9.0,<3.10.0" "albumentations" "opencv-python-headless" "gTTS" "pyttsx3" "qrcode[pil]" "sentence-transformers" "faiss-cpu" "edge-tts" "rich<14.0.0"', shell=True, check=True)
subprocess.run("apt-get install -y ffmpeg", shell=True, check=False)
print("   ✅ 所有依賴安裝完成！")

# %%
# ============================================================================
# STEP 5: 啟動主程式
# ============================================================================
print("\n[6/7] 系統啟動...")
from huggingface_hub import login
if hf_token:
    try:
        login(token=hf_token)
        print("   ✅ Hugging Face Login Success")
    except Exception as e:
        print(f"   ❌ Login Failed: {e}")
else:
    print("   ⚠️ Skipping Login (No Token).")

# ============================================================================
# 🔥 PHASE 1: V16 超擬真數據生成 (Impact Challenge Edition)
# ============================================================================
print("\n" + "=" * 80)
print("🎨 PHASE 1: V16 Hyper-Realistic Data Generation")
print("=" * 80)

import glob

# [終極神級修正] 全域動態雷達 (Omni-Radar)：無視 Kaggle 資料夾命名
print("🔍 啟動全域雷達掃描 V17 資料集...")
v17_train_json = None
# 1. 優先暴力掃描整個 /kaggle/input/ 目錄
kaggle_v17 = glob.glob("/kaggle/input/**/dataset_v17_train.json", recursive=True)
if kaggle_v17:
    v17_train_json = kaggle_v17[0]
else:
    # 2. 備用：掃描本地工作目錄
    local_v17 = glob.glob("./**/dataset_v17_train.json", recursive=True)
    if local_v17:
        v17_train_json = local_v17[0]

if v17_train_json:
    print(f"⏩ V17 Dataset already exists at {v17_train_json}")
    print("   Skipping generation to save time (Speed Run Mode Active)...")
else:
    print("🏭 Generating V17 Dataset (3D Pills + QR Codes + Human Touch)...")
    try:
        # [Omni-Nexus Fix] 執行正確的 V17 生成器
        subprocess.run(["python", "generate_v17_fusion.py"], check=True)
        print("✅ V17 Dataset Generation Complete!")
    except Exception as e:
        print(f"⚠️ V17 Generation Failed: {e}")
        print("   Falling back to V8 internal generator...")

# ============================================================================
# 🔥 PHASE 2: Stress Test 生成 (用於推論測試)
# ============================================================================
print("\n" + "=" * 80)
print("🧪 PHASE 2: Stress Test Generation (Inference Demo)")
print("=" * 80)

print("🔍 啟動全域雷達掃描壓力測試集...")
stress_test_dir = None
kaggle_stress = glob.glob("/kaggle/input/**/stress_test_labels.json", recursive=True)
if kaggle_stress:
    stress_test_dir = os.path.dirname(kaggle_stress[0])
else:
    local_stress = glob.glob("./**/stress_test_labels.json", recursive=True)
    if local_stress:
        stress_test_dir = os.path.dirname(local_stress[0])

if stress_test_dir:
    print(f"⏩ Stress Test already exists at {stress_test_dir}")
    print("   Skipping generation...")
else:
    print("🔥 Generating Stress Test Cases (Edge Case Validation)...")
    try:
        subprocess.run(["python", "generate_stress_test.py"], check=True)
        print("✅ Stress Test Generation Complete!")
    except Exception as e:
        print(f"⚠️ Stress Test Generation Failed: {e}")

# ============================================================================
# 🔥 PHASE 3: 執行主程式 (V8 Training + Inference)
# ============================================================================
print("\n" + "=" * 80)
print("🧠 PHASE 3: Launching SilverGuard V8 Training Pipeline")
print("=" * 80)

# 設定環境變數，讓 V8 使用 V17 數據
v17_found = False
if v17_train_json:
    v17_image_dir = os.path.dirname(v17_train_json)
    try:
        # 檢查該路徑下是否有足夠的圖片
        image_count = len([f for f in os.listdir(v17_image_dir) if f.endswith('.png')])
        if image_count > 100:
            os.environ["MEDGEMMA_USE_V17_DATA"] = "1"
            os.environ["MEDGEMMA_V17_DIR"] = os.path.abspath(v17_image_dir)  # 使用絕對路徑
            print(f"✅ V8 will use V17 Hyper-Realistic Dataset ({image_count} images from {v17_image_dir})")
            v17_found = True
        else:
            print(f"⚠️ Found V17 dir at {v17_image_dir} but only {image_count} images (need >100)")
    except Exception as e:
        print(f"⚠️ Error checking {v17_image_dir}: {e}")

if not v17_found:
    os.environ["MEDGEMMA_USE_V17_DATA"] = "0"
    print("⚠️ V8 will use internal V5 generator (V17 dir not found in any location)")

# 執行主程式 (註解說明：請在 Notebook 的下一個 Cell 手動執行 !python agent_engine.py，避免 Bootstrap 卡死)
# subprocess.run(["python", "agent_engine.py"], check=True)
print("🎉 Bootstrap Complete! Now run app.py in a separate cell.")
