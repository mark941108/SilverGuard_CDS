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
        subprocess.run("cp *.py SilverGuard/", shell=True, check=True, stderr=subprocess.DEVNULL)
    except:
        pass
    
else:
    # 【場景 B】乾淨環境 -> 從 GitHub 拉取
    print("   ☁️ 未偵測到本地檔案，啟動 [GitHub Clone Mode]...")
    import shutil
    if os.path.exists("SilverGuard"):
        shutil.rmtree("SilverGuard")
    
    # [FIX] 防止 Git Auth 卡死 (The Silent Hang Fix)
    # 只有在真的有 token 時才加入 @，否則 Git 會跳出隱形密碼輸入框導致卡死
    if gh_token:
        repo_url = f"https://{gh_token}@github.com/mark941108/SilverGuard.git"
    else:
        print("   ⚠️ 無 GitHub Token，嘗試 Public Clone (無密碼模式)...")
        repo_url = "https://github.com/mark941108/SilverGuard.git"
        
    import subprocess
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
else:
    print("   ℹ️ 已經在 SilverGuard 目錄內，略過切換。")

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
subprocess.run("apt-get update -y && apt-get install -y libespeak1 libsndfile1 ffmpeg", shell=True, check=True)

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

# 4. Hugging Face Stack (升級支援 Gemma 3)
# 原因: Gemma 3 架構需要最新版 Transformers (>=4.51.0)
# 修正: 不再鎖定 4.47.1，改為安裝最新穩定版
# ⚠️ [Omni-Nexus Warning] Version Roulette: transformers 5.0+ may introduce breaking changes.
# Update with caution! Currently unpinned to support checking for latest versions.
print("   ⬇️ 安裝 Hugging Face Stack (Gemma 3 Support)...")
subprocess.run('pip install -U "huggingface-hub>=0.29.0" "transformers>=4.51.0" accelerate bitsandbytes peft datasets', shell=True, check=True)

# 5. 應用層依賴 (RAG, Vision, Audio)
print("   ⬇️ 安裝應用層依賴...")
subprocess.run("pip install -U sentence-transformers faiss-cpu pydub", shell=True, check=True)
subprocess.run("pip install -U pillow==10.4.0 librosa soundfile", shell=True, check=True)
subprocess.run("pip install -U qrcode[pil] albumentations==1.3.1 opencv-python-headless gTTS edge-tts nest_asyncio pyttsx3", shell=True, check=True)
subprocess.run("pip install -U gradio==4.44.1", shell=True, check=True)

print("   ✅ 所有依賴安裝完成！")

# %%
# ============================================================================
# STEP 5: 啟動主程式
# ============================================================================
print("\n[6/7] 系統啟動...")

from huggingface_hub import login

# [Omni-Nexus Fix] Safe Login Strategy
if not hf_token:
    print("\n⚠️ WARNING: HUGGINGFACE_TOKEN is missing!")
    print("   MedGemma requires a token usually. attempting manual input (or press Enter to skip).")
    try:
        # In Kaggle non-interactive mode this might fail, so we wrap it
        manual_input = input("🔑 Please paste your HF Token here: ").strip()
        if manual_input:
            hf_token = manual_input
    except:
        print("   (Input skipped/failed)")

if hf_token:
    try:
        login(token=hf_token)
        print("   ✅ Hugging Face Login Success")
    except Exception as e:
        print(f"   ❌ Login Failed: {e}")
        print("   ➡️ Continuing anyway... (Public weights might work)")
else:
    print("   ⚠️ Skipping Login (No Token). Verification may fail for Gated Models.")

print("\n" + "=" * 80)
print("🚀 啟動 SilverGuard: Impact Research Edition (V12.13 Gemma 3 Fix)")
print("=" * 80)

# ============================================================================
# 🔥 PHASE 1: V16 超擬真數據生成 (Impact Challenge Edition)
# ============================================================================
print("\n" + "=" * 80)
print("🎨 PHASE 1: V16 Hyper-Realistic Data Generation")
print("=" * 80)

# Check if V16 data already exists (skip if running multiple times)
import os
# [Omni-Nexus Fix] 更新路徑至 V17
v17_train_json = "./assets/lasa_dataset_v17_compliance/dataset_v17_train.json"

if os.path.exists(v17_train_json):
    print(f"⏩ V17 Dataset already exists at {v17_train_json}")
    print("   Skipping generation to save time...")
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

stress_test_dir = "./assets/stress_test"
if os.path.exists(stress_test_dir) and len(os.listdir(stress_test_dir)) > 0:
    print(f"⏩ Stress Test already exists at {stress_test_dir}")
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

# 設定環境變數，讓 V8 使用 V16 數據
# 設定環境變數，讓 V8 使用 V17 數據
if os.path.exists(v17_train_json):
    os.environ["MEDGEMMA_USE_V17_DATA"] = "1"
    os.environ["MEDGEMMA_V17_DIR"] = "./assets/lasa_dataset_v17_compliance"
    print("✅ V8 will use V17 Hyper-Realistic Dataset")
else:
    os.environ["MEDGEMMA_USE_V17_DATA"] = "0"
    print("⚠️ V8 will use internal V5 generator (fallback)")

# 執行主程式
subprocess.run(["python", "agent_engine.py"], check=True)

