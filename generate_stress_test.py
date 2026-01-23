"""
Gallery of Horrors - Stress Test Generator (V9: 2026 Flagship Edition)
======================================================================
Designed for MedGemma Impact Challenge - "Agentic Workflow Prize"
Compliance: Taiwan Pharmacist Act (13 Items) + 2026 Elderly Friendly UX.

Features:
1.  Visual Timing: Full Bowl (After Meal) vs Empty Bowl (Before Meal).
2.  Layout: Red Hotline (Top), Big Patient Name (Left), Pill Photo (Right).
3.  Safety: Anti-confusion Color Bands, Warning Icons (No Drive/Alcohol).
4.  Physical: Simulated Hole Punch (Wall hanging).
"""

import os
import random
import qrcode
import math
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Output Config
OUTPUT_DIR = "assets/stress_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)
IMG_WIDTH = 1200  # 加寬以容納更清楚的圖示
IMG_HEIGHT = 1400 # 加高以容納底部完整資訊

# ==========================================
# 1. 資源準備 (Auto-Font)
# ==========================================
FONT_URL = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf"
FONT_PATH = "NotoSansCJKtc-Regular.otf"

def get_font(size):
    if not os.path.exists(FONT_PATH):
        try:
            print(f"⬇️ 下載中文字體中... ({size}px)")
            r = requests.get(FONT_URL)
            with open(FONT_PATH, "wb") as f:
                f.write(r.content)
        except:
            return ImageFont.load_default()
    return ImageFont.truetype(FONT_PATH, size)

# ==========================================
# 2. 2026 進階圖示引擎 (Advanced Pictograms)
# ==========================================

def draw_sun(draw, x, y, size, color="black"):
    """ 太陽 (實心/空心) """
    cx, cy = x, y
    r = size // 3
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=color, width=3)
    for i in range(0, 360, 45):
        rad = math.radians(i)
        sx = cx + math.cos(rad) * (r+4)
        sy = cy + math.sin(rad) * (r+4)
        ex = cx + math.cos(rad) * (size//1.5)
        ey = cy + math.sin(rad) * (size//1.5)
        draw.line([sx, sy, ex, ey], fill=color, width=3)

def draw_moon(draw, x, y, size):
    """ 月亮 + 星星 """
    draw.chord([x-size//2, y-size//2, x+size//2, y+size//2], start=30, end=330, outline="black", width=3)
    sx, sy = x - 10, y
    draw.line([sx-5, sy, sx+5, sy], fill="black", width=2)
    draw.line([sx, sy-5, sx, sy+5], fill="black", width=2)

def draw_bowl_full(draw, x, y, size):
    """ 盛滿飯的碗 (飯後) """
    # 碗
    draw.chord([x-size//2, y-size//4, x+size//2, y+size//2], start=0, end=180, outline="black", width=3)
    draw.line([x-size//4, y+size//2, x+size//4, y+size//2], fill="black", width=3)
    # 飯 (堆高高)
    draw.arc([x-size//2+2, y-size//2, x+size//2-2, y], start=180, end=0, fill="black", width=3)
    # 筷子
    draw.line([x+size//4, y-size//2, x+size//2, y+size//4], fill="black", width=3)

def draw_bowl_empty(draw, x, y, size):
    """ 空碗 (飯前) """
    # 碗
    draw.chord([x-size//2, y-size//4, x+size//2, y+size//2], start=0, end=180, outline="black", width=3)
    draw.line([x-size//4, y+size//2, x+size//4, y+size//2], fill="black", width=3)
    # 筷子平放
    draw.line([x-size//2, y-size//2, x+size//2, y-size//2], fill="black", width=2)

def draw_bed(draw, x, y, size):
    """ 床鋪 """
    draw.rectangle([x-size//2, y, x+size//2, y+size//4], outline="black", width=3)
    draw.rectangle([x-size//2, y-10, x-size//2+15, y], fill="black") # 枕頭
    # Zzz
    f = get_font(20)
    draw.text((x, y-40), "Zzz", fill="black", font=f)

def draw_warning_icon(draw, x, y, size, type="car"):
    """ 警示圖標 (禁止開車/飲酒) """
    draw.ellipse([x-size//2, y-size//2, x+size//2, y+size//2], outline="red", width=4)
    draw.line([x-size//2.5, y+size//2.5, x+size//2.5, y-size//2.5], fill="red", width=4)
    
    if type == "car":
        draw.rectangle([x-15, y-5, x+15, y+10], fill="black") # 車身
        draw.ellipse([x-12, y+10, x-5, y+18], fill="black") # 輪
        draw.ellipse([x+5, y+10, x+12, y+18], fill="black")
    elif type == "wine":
        draw.polygon([(x-8, y-10), (x+8, y-10), (x, y+5)], outline="black", width=2)
        draw.line([x, y+5, x, y+15], fill="black", width=2)

def draw_indication_icon(draw, x, y, size, type="heart"):
    """ 適應症圖示 """
    if type == "heart":
        draw.polygon([(x, y+15), (x-15, y-5), (x, y-15), (x+15, y-5)], fill="red")
    elif type == "stomach":
        draw.arc([x-15, y-15, x+15, y+15], start=30, end=270, fill="gray", width=3)

# ==========================================
# 3. 核心組件 (Layout Components)
# ==========================================

def draw_pill_photo_sim(draw, x, y, drug):
    """ 1:1 藥物外觀照片模擬 (Pseudo-3D) """
    # 背景相紙感
    draw.rectangle([x, y, x+200, y+150], fill=(240, 240, 240), outline="gray", width=1)
    draw.text((x+10, y+5), "藥品真實外觀 (Size 1:1)", fill="gray", font=get_font(20))
    
    cx, cy = x + 100, y + 85
    size = 80 # 大尺寸
    
    # 陰影 (Shadow)
    draw.ellipse([cx-size//2+5, cy-size//2+5, cx+size//2+5, cy+size//2+5], fill=(200,200,200))
    
    colors = {"white": (255,255,255), "yellow": (255,240,180), "pink": (255,200,200)}
    fill = colors.get(drug['color'], (255,255,255))
    
    if drug['shape'] == 'circle':
        draw.ellipse([cx-size//2, cy-size//2, cx+size//2, cy+size//2], fill=fill, outline="black", width=2)
    elif drug['shape'] == 'oval':
        draw.ellipse([cx-size//1.2, cy-size//2, cx+size//1.2, cy+size//2], fill=fill, outline="black", width=2)
        
    # 刻痕與光澤
    draw.line([cx-20, cy, cx+20, cy], fill=(200,200,200), width=2)
    draw.arc([cx-size//4, cy-size//4, cx, cy], start=180, end=270, fill="white", width=3) # 反光

def draw_usage_grid_2026(draw, x, y, w, h, drug):
    """ 2026 旗艦版用法表格 """
    # 外框
    draw.rectangle([x, y, x+w, y+h], outline="black", width=4)
    col_w = w // 4
    for i in range(1, 4):
        draw.line([x+i*col_w, y, x+i*col_w, y+h], fill="black", width=2)
        
    headers = ["早上", "中午", "晚上", "睡前"]
    # 用法解析
    usage_code = drug['usage']
    timing = drug['timing'] # 飯前 or 飯後
    
    targets = []
    if "BID" in usage_code: targets = [0, 2]
    elif "TID" in usage_code: targets = [0, 1, 2]
    elif "QD" in usage_code: targets = [0]
    elif "QN" in usage_code: targets = [3]
    
    for i in range(4):
        bx = x + i*col_w
        cx = bx + col_w//2
        cy = y + h//2
        
        # 1. 標題 (大字)
        draw.text((bx+15, y+10), headers[i], fill="black", font=get_font(28))
        
        # 2. 時間圖示 (Sun/Moon)
        icon_y = cy - 40
        if i == 0: draw_sun(draw, cx, icon_y, 40)
        elif i == 1: draw_sun(draw, cx, icon_y, 40)
        elif i == 2: draw_sun(draw, cx, icon_y, 40, "gray") # 傍晚
        elif i == 3: draw_moon(draw, cx, icon_y, 40)
        
        # 3. 飯碗圖示 (Before/After Meal)
        # 只有在「要吃」的那個時段才顯示碗，減少視覺干擾
        if i in targets and i != 3: # 睡前通常不吃飯
            bowl_y = icon_y + 40
            if "飯後" in timing:
                draw_bowl_full(draw, cx+30, bowl_y, 30)
            else:
                draw_bowl_empty(draw, cx+30, bowl_y, 30)
        
        # 4. 數量確認 (Big Red Circle)
        if i in targets:
            draw.ellipse([cx-30, cy+20, cx+30, cy+80], outline="red", width=5)
            draw.text((cx-12, cy+25), "1", fill="red", font=get_font(40))
        else:
            # 淡化處理
            draw.line([cx-20, cy+40, cx+20, cy+60], fill="lightgray", width=3)
            draw.line([cx-20, cy+60, cx+20, cy+40], fill="lightgray", width=3)

def apply_texture(img):
    overlay = Image.new("RGBA", img.size, (255, 252, 240, 20))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    arr = np.array(img)
    noise = np.random.normal(0, 3, arr.shape).astype(np.uint8)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

# ==========================================
# 4. 主生成器 (Main Pipeline)
# ==========================================

def generate_v9_bag(filename, patient, drug, is_danger=False):
    img = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), "white")
    draw = ImageDraw.Draw(img)
    
    # Fonts
    f_h1 = get_font(50) # 機構
    f_h2 = get_font(40) # 重點標題
    f_body = get_font(28)
    f_huge = get_font(60) # 藥名
    f_warn = get_font(32)

    # --- 1. Top Header (機構、紅字專線、QR) ---
    # [法定 9] 機構名稱
    draw.text((50, 40), "MedGemma 聯合醫療體系", fill="black", font=f_h1)
    # [2026] 服務專線 (大紅字)
    draw.text((50, 100), "用藥諮詢專線: (02) 2345-6789", fill="red", font=f_h2)
    
    # [2026] QR Code (Top Right)
    qr = qrcode.QRCode(box_size=5, border=2)
    qr.add_data(f"https://medgemma.tw/verify?id={drug['id']}")
    qr_img = qr.make_image(fill_color="black", back_color="white")
    img.paste(qr_img, (IMG_WIDTH-180, 30))
    draw.text((IMG_WIDTH-180, 160), "語音朗讀", fill="black", font=get_font(20))
    
    draw.line([(30, 190), (IMG_WIDTH-30, 190)], fill="black", width=5)

    # --- 2. Center Left: Patient Info (Big Font) ---
    y_p = 220
    # [法定 1] 姓名 (Huge)
    draw.text((50, y_p), f"姓名: {patient['name']}", fill="black", font=f_h1)
    # [法定 2] 性別
    draw.text((400, y_p+15), f"{patient['gender']}", fill="black", font=f_h2)
    # [法定 12] 調劑日期
    draw.text((50, y_p+70), f"調劑日: 115/01/22", fill="black", font=f_body)
    # [法定] 病歷號
    draw.text((400, y_p+70), f"病歷號: {random.randint(100000,999999)}", fill="black", font=f_body)

    # --- 3. Center Right: Pill Photo (1:1) ---
    # [2026] 藥物外觀照片
    draw_pill_photo_sim(draw, 800, y_p, drug)

    # --- 4. Drug Core Info (Color Coding) ---
    y_drug = 400
    # [2026] 顏色標記 (左側色條)
    color_map = {"高血壓": "green", "糖尿病": "orange", "失眠": "blue"}
    bar_color = color_map.get(drug['cat'], "gray")
    draw.rectangle([20, y_drug, 40, y_drug+150], fill=bar_color)
    
    # [法定 3] 藥名 (Huge Blue)
    draw.text((60, y_drug), drug['cht'], fill="blue", font=f_huge)
    draw.text((60, y_drug+70), drug['eng'], fill="black", font=f_h2)
    
    # [法定 7] 適應症圖示
    draw.text((60, y_drug+120), f"適應症: {drug['indication']}", fill="black", font=f_h2)
    if "心" in drug['indication']: draw_indication_icon(draw, 400, y_drug+135, 30, "heart")
    
    # [法定 4, 5] 劑量
    dose_val = "5000mg" if is_danger else drug['dose']
    if is_danger: draw.text((600, y_drug+120), "⚠️劑量異常", fill="red", font=f_warn)
    draw.text((600, y_drug), f"劑量: {dose_val}", fill="black", font=f_h2)
    draw.text((600, y_drug+50), "總量: 28 顆", fill="black", font=f_h2)

    # --- 5. Usage Grid (The Main Feature) ---
    y_grid = 600
    # [法定 6] 用法 (Big Pictograms)
    draw_usage_grid_2026(draw, 50, y_grid, 1100, 200, drug)
    
    # [法定] 備註
    draw.text((50, y_grid+210), f"備註: {drug['timing']} 服用", fill="black", font=f_h2)

    # --- 6. Warnings & Footer ---
    y_warn = 880
    # [法定 8] 警語
    draw.rectangle([50, y_warn, 1150, y_warn+180], fill=(255, 245, 245), outline="red", width=3)
    draw.text((70, y_warn+10), "⚠️ 安全警語 / 副作用:", fill="red", font=f_warn)
    draw.text((70, y_warn+60), drug['warning'], fill="red", font=f_h2)
    
    # 警示圖標
    if "開車" in drug['warning']: draw_warning_icon(draw, 1000, y_warn+90, 60, "car")
    if "酒" in drug['warning']: draw_warning_icon(draw, 1100, y_warn+90, 60, "wine")

    # [2026] 防呆打孔 (左側圓圈)
    draw.ellipse([10, 650, 30, 670], outline="gray", width=2)
    draw.ellipse([10, 750, 30, 770], outline="gray", width=2)

    # Footer (法定 10, 11, 13)
    y_foot = 1100
    draw.line([(30, y_foot), (IMG_WIDTH-30, y_foot)], fill="gray", width=2)
    draw.text((50, y_foot+20), "【三核對】: □ 姓名正確  □ 外觀相符  □ 用法清楚", fill="black", font=f_h2)
    draw.text((50, y_foot+80), "調劑藥師: 王大明  |  核對藥師: 李小美  |  地址: 台北市...", fill="gray", font=f_body)

    # Texture
    img = apply_texture(img)
    img.save(filename)
    print(f"✅ V9 旗艦版生成完畢: {filename}")

# Database
PATIENTS = [{"name": "林罔市", "gender": "女", "born": 28}, {"name": "陳進財", "gender": "男", "born": 32}]
DRUGS = [
    {"id": "MET", "cht": "美福明降血糖片", "eng": "Metformin", "dose": "500mg", "cat": "糖尿病", "color": "white", "shape": "circle", "usage": "BID", "timing": "飯後", "warning": "服用後禁止飲酒，若有腹痛請就醫", "indication": "糖尿病控制"},
    {"id": "AML", "cht": "脈優降壓錠", "eng": "Amlodipine", "dose": "5mg", "cat": "高血壓", "color": "yellow", "shape": "oval", "usage": "QD", "timing": "飯後", "warning": "避免食用葡萄柚", "indication": "高血壓/心臟"},
    {"id": "EST", "cht": "悠樂丁錠", "eng": "Estazolam", "dose": "2mg", "cat": "失眠", "color": "white", "shape": "circle", "usage": "QN", "timing": "睡前", "warning": "服用後禁止開車，有嗜睡風險", "indication": "失眠輔助"}
]

if __name__ == "__main__":
    print("🏥 啟動 V9 2026 旗艦版生成引擎 (Legal + UX + Digital)...")
    for i in range(1, 6):
        p = random.choice(PATIENTS)
        d = random.choice(DRUGS)
        generate_v9_bag(f"{OUTPUT_DIR}/taiwan_v9_flagship_{i}.jpg", p, d, is_danger=(i==5))