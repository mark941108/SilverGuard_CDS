"""
九宮格證據圖自動生成器 (Evidence Grid Generator)
====================================================
用途: 將 Sim2Physical 測試的 10 張照片拼成專業級證據圖
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_evidence_grid(photo_dir, output_path):
    """
    建立 4x3 九宮格證據圖
    
    Args:
        photo_dir: 包含 IMG_0001.jpg ~ IMG_0010.jpg 的目錄
        output_path: 輸出檔案路徑（例如 sim2physical_evidence_grid.jpg）
    """
    
    # 載入 10 張照片
    photos = []
    for i in range(1, 11):
        photo_path = os.path.join(photo_dir, f"IMG_{i:04d}.jpg")
        if not os.path.exists(photo_path):
            print(f"⚠️  找不到 {photo_path}，使用佔位符")
            # 建立灰色佔位符
            placeholder = Image.new('RGB', (400, 400), (200, 200, 200))
            photos.append(placeholder)
        else:
            photo = Image.open(photo_path)
            # 調整為統一尺寸
            photo_resized = photo.resize((400, 400), Image.Resampling.LANCZOS)
            photos.append(photo_resized)
    
    # 建立 4x3 網格（10 張圖 + 2 個標題位）
    grid_width = 4 * 400
    grid_height = 3 * 400
    grid = Image.new('RGB', (grid_width, grid_height), 'white')
    
    # 貼上照片
    for idx, photo in enumerate(photos):
        row = idx // 4
        col = idx % 4
        x = col * 400
        y = row * 400
        grid.paste(photo, (x, y))
        
        # 加入編號標籤
        draw = ImageDraw.Draw(grid)
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        # 左上角標註 Test 編號
        label = f"Test {idx+1}"
        draw.rectangle([x+5, y+5, x+90, y+35], fill='black')
        draw.text((x+10, y+10), label, fill='white', font=font)
    
    # 加入標題（在第 11-12 格位置）
    draw = ImageDraw.Draw(grid)
    try:
        title_font = ImageFont.truetype("arial.ttf", 32)
        label_font = ImageFont.truetype("arial.ttf", 20)
    except:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    
    # 標題文字
    title_text = "SilverGuard Sim2Physical Validation"
    subtitle_text = "10-Point Optical Robustness Test"
    
    # 在右下角 2 格繪製標題
    title_x = 2 * 400 + 50
    title_y = 2 * 400 + 100
    
    draw.text((title_x, title_y), title_text, fill='black', font=title_font)
    draw.text((title_x, title_y + 50), subtitle_text, fill='gray', font=label_font)
    
    # 加入測試分類標註
    categories = [
        "Row 1: Baseline (0°, Natural Light)",
        "Row 2: Angle Stress (15°, 30°) + Lighting",
        "Row 3: Safety Mechanism Tests (Glare, Blur)"
    ]
    
    for i, cat in enumerate(categories):
        draw.text((10, i * 400 + grid_height - 90), cat, fill='blue', font=label_font)
    
    # 儲存
    grid.save(output_path, quality=95)
    print(f"✅ 九宮格證據圖已生成: {output_path}")
    print(f"   尺寸: {grid_width} x {grid_height}")

if __name__ == "__main__":
    import sys
    
    # 使用範例
    if len(sys.argv) > 1:
        photo_dir = sys.argv[1]
    else:
        photo_dir = "."  # 當前目錄
    
    output_path = "sim2physical_evidence_grid.jpg"
    
    print("📸 九宮格證據圖生成器")
    print(f"   來源目錄: {photo_dir}")
    print(f"   輸出檔案: {output_path}")
    print()
    
    create_evidence_grid(photo_dir, output_path)
    
    print()
    print("🎯 下一步:")
    print("   1. 查看生成的 sim2physical_evidence_grid.jpg")
    print("   2. 將此圖加入 README.md 的 Validation 章節")
    print("   3. 在報告中使用「安全機制」框架解讀結果")
