# 🧹 SilverGuard Asset Cleanup Script
# =====================================
# 此腳本會清除所有舊的訓練資料和生成的檔案，確保全新的 PNG 生成

## ⚠️ 警告
# 執行此腳本會刪除以下目錄：
# - assets/lasa_dataset_v17_compliance (V16 生成器輸出)
# - assets/lasa_dataset_v16_samples (舊的手動示範檔案)
# - assets/stress_test (壓力測試輸出)
# - medgemma_training_data_v5 (V5 內建生成器輸出)

# 請確認您已經備份任何需要保留的檔案！

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🐧 Kaggle / Linux 環境
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 方法 1: 直接在 Notebook Cell 中執行
```python
import shutil
import os

dirs_to_clean = [
    "assets/lasa_dataset_v17_compliance",
    "assets/lasa_dataset_v16_samples",
    "assets/stress_test",
    "medgemma_training_data_v5"
]

for directory in dirs_to_clean:
    if os.path.exists(directory):
        print(f"🗑️  Removing {directory}...")
        shutil.rmtree(directory)
        print(f"   ✅ Deleted")
    else:
        print(f"   ⏭️  {directory} does not exist (skipped)")

print("\n✨ Cleanup complete! Ready for fresh PNG generation.")
```

# 方法 2: Bash 指令 (Kaggle Terminal)
```bash
#!/bin/bash
echo "🧹 Starting cleanup..."

rm -rf assets/lasa_dataset_v17_compliance
echo "  ✅ Removed V17 compliance dataset"

rm -rf assets/lasa_dataset_v16_samples
echo "  ✅ Removed V16 samples"

rm -rf assets/stress_test
echo "  ✅ Removed stress test data"

rm -rf medgemma_training_data_v5
echo "  ✅ Removed V5 training data"

echo "✨ Cleanup complete!"
```

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🪟 Windows 本地環境 (PowerShell)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 方法 1: 在 PowerShell 中執行
```powershell
# 安全刪除（會先檢查是否存在）
$dirsToClean = @(
    "assets\lasa_dataset_v17_compliance",
    "assets\lasa_dataset_v16_samples",
    "assets\stress_test",
    "medgemma_training_data_v5"
)

foreach ($dir in $dirsToClean) {
    if (Test-Path $dir) {
        Write-Host "🗑️  Removing $dir..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $dir
        Write-Host "   ✅ Deleted" -ForegroundColor Green
    } else {
        Write-Host "   ⏭️  $dir does not exist (skipped)" -ForegroundColor Gray
    }
}

Write-Host "`n✨ Cleanup complete! Ready for fresh PNG generation." -ForegroundColor Cyan
```

# 方法 2: 單行指令（快速執行）
```powershell
Remove-Item -Recurse -Force assets\lasa_dataset_v17_compliance, assets\lasa_dataset_v16_samples, assets\stress_test, medgemma_training_data_v5 -ErrorAction SilentlyContinue
```

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📋 清理後的驗證步驟
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 1. 確認目錄已清空
```bash
# Linux/Kaggle
ls -la assets/
ls -la medgemma_training_data_v5/

# Windows PowerShell
Get-ChildItem assets\
Get-ChildItem medgemma_training_data_v5\
```

預期輸出應該只顯示其他未刪除的檔案（如 `hero_image.jpg` 等靜態資源）。

## 2. 重新執行生成器
```bash
# Kaggle: 直接 Run All
# 本地: 
python generate_v16_fusion.py
python generate_stress_test.py
```

## 3. 驗證新檔案格式
```bash
# Linux/Kaggle
ls assets/lasa_dataset_v17_compliance/*.png | head -5

# Windows PowerShell
Get-ChildItem assets\lasa_dataset_v17_compliance\*.png | Select-Object -First 5
```

預期輸出應該只顯示 `.png` 檔案，例如：
```
SOUND_ALIKE_CRITICAL_Norvasc_V000.png
SOUND_ALIKE_CRITICAL_Norvasc_V001.png
SOUND_ALIKE_CRITICAL_Norvasc_V002.png
...
```

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🚨 緊急回復 (Emergency Rollback)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

如果清理後發現有問題，可以從 GitHub 重新 clone：

```bash
# Kaggle
!rm -rf SilverGuard
!git clone --depth 1 https://{GITHUB_TOKEN}@github.com/mark941108/SilverGuard.git
cd SilverGuard

# 本地
cd "c:\Users\USER\Desktop\The MedGemma Impact Challenge"
git pull origin main
```

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📊 清理統計 (估計)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

預期刪除的檔案數量：
- V17 compliance: ~50 JPG + 50 JSON = 100 檔案
- V16 samples: ~10 JPG + 10 JSON = 20 檔案
- Stress test: ~16 JPG + metadata = 20 檔案
- V5 training data: 600 PNG + 2 JSON = 602 檔案

**總計**: 約 742 檔案 (~2.5 GB)

清理後將釋放磁碟空間，為新的 PNG 生成騰出空間。

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ✅ 安全提示
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. **備份重要檔案**: 如果您手動編輯過任何 JSON 或圖片，請先備份。
2. **檢查 Git 狀態**: 確保沒有未提交的變更會被誤刪。
3. **逐步執行**: 如果不確定，可以先刪除一個目錄，確認無誤再刪除其他。

**現在可以安全執行清理了！** 🚀
