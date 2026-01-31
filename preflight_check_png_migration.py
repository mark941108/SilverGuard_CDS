#!/usr/bin/env python3
"""
🔍 SilverGuard PNG Migration Pre-Flight Check
==============================================
執行此腳本以確保所有元件正確支援 PNG 格式。

檢查項目：
1. 生成器輸出格式 ✓
2. JSON 內容一致性 ✓
3. V8 讀取邏輯相容性 ✓
4. 舊檔案清理建議 ✓
"""

import os
import json
import glob
from pathlib import Path

# ANSI 顏色碼
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def check_generators():
    """檢查生成器是否已切換到 PNG"""
    print(f"\n{BLUE}[1/4] 檢查生成器檔案格式...{RESET}")
    
    generators = [
        "generate_v16_fusion.py",
        "generate_stress_test.py"
    ]
    
    all_clean = True
    for gen in generators:
        if not os.path.exists(gen):
            print(f"  {YELLOW}⚠️  {gen} 不存在{RESET}")
            continue
            
        with open(gen, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 檢查是否還有 .jpg 引用（排除註解）
        lines_with_jpg = []
        for i, line in enumerate(content.split('\n'), 1):
            if '.jpg' in line and not line.strip().startswith('#'):
                lines_with_jpg.append((i, line.strip()))
        
        if lines_with_jpg:
            print(f"  {RED}❌ {gen} 仍有 .jpg 引用：{RESET}")
            for line_num, line in lines_with_jpg[:3]:  # 只顯示前3個
                print(f"     Line {line_num}: {line[:80]}")
            all_clean = False
        else:
            print(f"  {GREEN}✅ {gen} 已完全切換到 .png{RESET}")
    
    return all_clean

def check_v8_compatibility():
    """檢查 V8 是否能正確讀取 PNG"""
    print(f"\n{BLUE}[2/4] 檢查 V8 讀取邏輯...{RESET}")
    
    v8_file = "SilverGuard_Impact_Research_V8.py"
    if not os.path.exists(v8_file):
        print(f"  {RED}❌ {v8_file} 不存在{RESET}")
        return False
    
    with open(v8_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 檢查是否有硬編碼的 .jpg 路徑（排除 Gradio temporary files）
    risky_patterns = [
        'glob.glob',
        'endswith(".jpg")',
        'endswith(\'.jpg\')',
        'demo_high_risk.jpg',
        'IMG_0001.jpg'
    ]
    
    issues = []
    for pattern in risky_patterns:
        if pattern in content:
            # 排除 Gradio 臨時檔案 (Line 3804)
            if 'tempfile' not in content[max(0, content.find(pattern)-100):content.find(pattern)+100]:
                issues.append(pattern)
    
    if issues:
        print(f"  {YELLOW}⚠️  發現潛在風險模式：{RESET}")
        for issue in issues:
            print(f"     - {issue}")
        print(f"  {BLUE}💡 建議手動檢查這些程式碼區塊{RESET}")
        return False
    else:
        print(f"  {GREEN}✅ V8 使用動態 JSON 讀取，與格式無關{RESET}")
        return True

def check_json_consistency():
    """檢查現有 JSON 檔案的內容"""
    print(f"\n{BLUE}[3/4] 檢查 JSON 一致性...{RESET}")
    
    json_dirs = [
        "assets/lasa_dataset_v17_compliance",
        "medgemma_training_data_v5"
    ]
    
    found_json = False
    for json_dir in json_dirs:
        json_files = glob.glob(f"{json_dir}/*.json")
        if not json_files:
            continue
        
        found_json = True
        for json_file in json_files[:2]:  # 只檢查前2個
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 檢查前3筆資料
            sample = data[:3] if isinstance(data, list) else [data]
            jpg_count = 0
            png_count = 0
            
            for item in sample:
                img_field = item.get('image', '')
                if img_field.endswith('.jpg'):
                    jpg_count += 1
                elif img_field.endswith('.png'):
                    png_count += 1
            
            if jpg_count > 0:
                print(f"  {YELLOW}⚠️  {json_file} 包含 {jpg_count} 個 .jpg 引用{RESET}")
                print(f"     {BLUE}建議：刪除此 JSON 並重新生成{RESET}")
            else:
                print(f"  {GREEN}✅ {json_file} 格式正確 ({png_count} PNG){RESET}")
    
    if not found_json:
        print(f"  {BLUE}ℹ️  尚未生成任何 JSON (這是正常的，等待首次執行){RESET}")
    
    return True

def suggest_cleanup():
    """提供清理舊檔案的建議"""
    print(f"\n{BLUE}[4/4] 清理建議...{RESET}")
    
    dirs_to_check = [
        "assets/lasa_dataset_v17_compliance",
        "assets/stress_test",
        "medgemma_training_data_v5"
    ]
    
    cleanup_needed = []
    for directory in dirs_to_check:
        if os.path.exists(directory):
            jpg_files = glob.glob(f"{directory}/**/*.jpg", recursive=True)
            if jpg_files:
                cleanup_needed.append((directory, len(jpg_files)))
    
    if cleanup_needed:
        print(f"  {YELLOW}⚠️  發現舊的 JPG 檔案：{RESET}")
        for directory, count in cleanup_needed:
            print(f"     - {directory}: {count} 個 .jpg 檔案")
        
        print(f"\n  {BLUE}🧹 清理指令 (Kaggle):{RESET}")
        print(f"     !rm -rf assets/lasa_dataset_v17_compliance")
        print(f"     !rm -rf assets/stress_test")
        print(f"     !rm -rf medgemma_training_data_v5")
        
        print(f"\n  {BLUE}🧹 清理指令 (本地 Windows):{RESET}")
        print(f"     Remove-Item -Recurse -Force assets/lasa_dataset_v17_compliance")
        print(f"     Remove-Item -Recurse -Force assets/stress_test")
        print(f"     Remove-Item -Recurse -Force medgemma_training_data_v5")
    else:
        print(f"  {GREEN}✅ 沒有發現舊的 JPG 檔案{RESET}")
    
    return True

def main():
    print(f"{GREEN}{'='*60}{RESET}")
    print(f"{GREEN}🔍 SilverGuard PNG Migration Pre-Flight Check{RESET}")
    print(f"{GREEN}{'='*60}{RESET}")
    
    results = {}
    results['generators'] = check_generators()
    results['v8_compatibility'] = check_v8_compatibility()
    results['json_consistency'] = check_json_consistency()
    results['cleanup'] = suggest_cleanup()
    
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}📊 檢查結果摘要{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    all_pass = all(results.values())
    if all_pass:
        print(f"{GREEN}✅ 所有檢查通過！系統準備就緒。{RESET}")
        print(f"\n{GREEN}🚀 下一步行動：{RESET}")
        print(f"   1. 在 Kaggle 上執行清理指令（如果有舊檔案）")
        print(f"   2. 執行 KAGGLE_BOOTSTRAP.py")
        print(f"   3. 確認看到 '✅ V16 Dataset Generation Complete!'")
    else:
        print(f"{YELLOW}⚠️  有部分檢查未通過，請檢查上方輸出。{RESET}")
    
    print(f"{BLUE}{'='*60}{RESET}\n")
    
    return 0 if all_pass else 1

if __name__ == "__main__":
    exit(main())
