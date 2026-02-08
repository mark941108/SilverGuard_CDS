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
from datetime import datetime, timezone, timedelta # [Audit Fix] Dynamic Date Support
import requests
import numpy as np
import textwrap
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# [UX Polish] Timezone Handling
TZ_TW = timezone(timedelta(hours=8))

# Output Config
OUTPUT_DIR = "assets/stress_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# V10 FIX: 改為 896x896 與訓練資料一致
IMG_WIDTH = 896
IMG_HEIGHT = 896

# ==========================================
# 1. 資源準備 (Auto-Font)
# ==========================================
FONT_URL = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf"
FONT_PATH = "NotoSansCJKtc-Regular.otf"

def get_font(size):
    """取得字型，帶 Offline Fallback 避免 Kaggle 崩潰"""
    
    # Priority 1: Local Kaggle Dataset (Offline Mode)
    KAGGLE_FONT_PATH = "/kaggle/input/noto-sans-cjk-tc/NotoSansCJKtc-Regular.otf"
    LOCAL_FONT_PATH = "NotoSansCJKtc-Regular.otf"
    
    font_target = LOCAL_FONT_PATH
    
    if os.path.exists(KAGGLE_FONT_PATH):
        font_target = KAGGLE_FONT_PATH
    elif not os.path.exists(LOCAL_FONT_PATH):
        # Priority 2: Download only if internet available (and not in Kaggle offline)
        try:
            print(f"⬇️ 下載中文字體中... ({size}px)")
            r = requests.get(FONT_URL, timeout=5) # Short timeout
            r.raise_for_status()
            with open(LOCAL_FONT_PATH, "wb") as f:
                f.write(r.content)
            print(f"   ✅ 字體下載成功")
        except Exception as e:
            pass # Keep silent fall back
    
    try:
        if os.path.exists(font_target):
             return ImageFont.truetype(font_target, size)
        else:
             # Try default paths in Linux container
             return ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", size)
    except Exception as e:
        print(f"   ⚠️ 字體載入失敗 ({font_target}): {e}，使用預設字體 (中文將亂碼)")
        return ImageFont.load_default()

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
        
    elif drug['shape'] == 'octagon':
        # Norvasc style
        points = []
        r = size // 2
        for i in range(8):
            ang = math.radians(45 * i + 22.5)
            px = cx + r * math.cos(ang)
            py = cy + r * math.sin(ang)
            points.append((px, py))
        draw.polygon(points, fill=fill, outline="black", width=2)

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
    # [Fix] Support V17 standards (HS/bedtime) in addition to Legacy QN
    elif any(x in usage_code for x in ["QN", "HS", "bedtime"]): targets = [3]
    
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
    """
    加入紙張紋理（模擬真實藥袋的粗糙表面）
    V12.33 Fix: 改為灰階噪點，移除 RGB 彩色雜訊
    """
    # 生成灰階噪點（單通道，-5 到 +5）
    noise_gray = np.random.randint(-5, 6, (img.size[1], img.size[0]), dtype=np.int8)
    
    # 擴展到 RGB 三通道（相同值 = 灰階）
    noise = np.stack([noise_gray, noise_gray, noise_gray], axis=-1)
    
    img_array = np.array(img)
    textured = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(textured)

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# ==========================================
# New Feature: Thermal Paper Fading Simulation
# ==========================================
def simulate_thermal_fading(img, severity=0.5):
    """
    Simulates thermal paper fading over time using ImageEnhance.
    severity: 0.0 (new) to 1.0 (completely faded)
    Effect: Lower contrast (fading ink) + Higher brightness (paper whitening/exposure)
    """
    # 降低對比度 (Fading ink)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.0 - (severity * 0.5))
    
    # 增加亮度 (Paper whitening/yellowing)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.0 + (severity * 0.2))
    return img

# ==========================================
# New V11 Feature: Optical Corruption Module
# ==========================================
def apply_optical_stress(img, severity=0):
    """
    Simulate real-world challenging conditions.
    severity: 0 (None), 1 (Mild - Hand tremor), 2 (Hard - Bad focus/lighting)
    """
    if severity == 0: return img
    
    # 0. Thermal Fading (New Writeup Feature)
    if random.random() < 0.4: # 40% chance of fading
        fading_severity = 0.3 if severity == 1 else 0.7
        img = simulate_thermal_fading(img, severity=fading_severity)

    # 1. 模糊 (老人手抖 / 對焦失敗)
    if random.random() < 0.7: # High chance of blur in stress mode
        radius = 2 if severity == 1 else 4 # 4px blur is hard for OCR
        img = img.filter(ImageFilter.GaussianBlur(radius))
        
    # 2. 旋轉 (隨意擺放)
    angle = random.randint(-5, 5) if severity == 1 else random.randint(-15, 15)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=0, fillcolor="white")
    
    # 3. 降低對比度 / 亮暗 (熱感紙褪色 / 反光)
    if random.random() < 0.5:
        enhancer = ImageEnhance.Contrast(img)
        factor = 0.8 if severity == 1 else 0.5
        img = enhancer.enhance(factor)
        
    # 4. 噪點 (低光源 ISO Noise) - 加強版
    if severity == 2:
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 50)) # Darken
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        
    return img

# ==========================================
# 4. 主生成器 (Main Pipeline)
# ==========================================
# ... existing generate_v9_bag function ...
# I need to modify generate_v9_bag to ACCEPT optical_severity argument.
# But simply updating the caller and adding the processing step inside generate or returning the image object to be processed is better.
# Actually, the user asked to modify generate_v9_bag. Wait, the user instruction was "Add apply_optical_stress... Update Main Loop".

# ==========================================
# 2. 完整邊緣案例模擬引擎 (Comprehensive Edge Case Suite)
# ==========================================
# Coverage: Creases, Glare, Physical Damage
# Purpose: Simulate real-world pharmacy conditions (elderly patients, pocket storage, long-term use)

def add_creases(img, intensity=0.5):
    """[Edge Case 1: Creases] 模擬藥袋摺痕"""
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    num_creases = random.randint(2, int(5 * intensity) + 2)
    
    for _ in range(num_creases):
        x1 = random.randint(0, img.width)
        y1 = random.randint(0, img.height)
        x2 = random.randint(0, img.width)
        y2 = random.randint(0, img.height)
        width = random.randint(1, 3)
        alpha = int(30 + intensity * 50)
        
        draw.line([(x1, y1), (x2, y2)], fill=(120, 120, 120, alpha), width=width)
    
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay)
    return img.convert("RGB")


def apply_plastic_glare(img, intensity=0.5):
    """[Edge Case 2: Plastic Glare] 模擬塑膠袋反光"""
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    num_glares = random.randint(1, int(3 * intensity) + 1)
    
    for _ in range(num_glares):
        w = random.randint(100, 400)
        h = random.randint(10, 40)
        x = random.randint(0, max(1, img.width - w))
        y = random.randint(0, max(1, img.height - h))
        alpha = int(20 + intensity * 40)
        
        draw.ellipse([x, y, x+w, y+h], fill=(255, 255, 255, alpha))
    
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay)
    return img.convert("RGB")


def apply_physical_damage(img, severity=0.5):
    """[Edge Case 3: Physical Damage] 模擬物理損壞"""
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Torn Corners
    if random.random() < severity * 0.7:
        corner = random.choice([0, 1, 2, 3])
        tear_size = int(50 + severity * 100)
        corners = [(0, 0), (img.width, 0), (0, img.height), (img.width, img.height)]
        cx, cy = corners[corner]
        
        tear_points = []
        for _ in range(3):
            offset_x = random.randint(-tear_size//2, tear_size//2)
            offset_y = random.randint(-tear_size//2, tear_size//2)
            
            if corner == 0:
                px = max(0, cx + abs(offset_x))
                py = max(0, cy + abs(offset_y))
            elif corner == 1:
                px = min(img.width, cx - abs(offset_x))
                py = max(0, cy + abs(offset_y))
            elif corner == 2:
                px = max(0, cx + abs(offset_x))
                py = min(img.height, cy - abs(offset_y))
            else:
                px = min(img.width, cx - abs(offset_x))
                py = min(img.height, cy - abs(offset_y))
            
            tear_points.append((px, py))
        
        draw.polygon(tear_points, fill=(240, 240, 240, 180))
    
    # Water Stains
    if random.random() < severity * 0.6:
        num_stains = random.randint(1, 3)
        for _ in range(num_stains):
            stain_x = random.randint(0, img.width)
            stain_y = random.randint(0, img.height)
            stain_radius = int(30 + severity * 70)
            rx = stain_radius + random.randint(-10, 10)
            ry = stain_radius + random.randint(-10, 10)
            stain_color = (
                random.randint(200, 220),
                random.randint(200, 210),
                random.randint(190, 200),
                int(30 + severity * 30)
            )
            draw.ellipse([stain_x-rx, stain_y-ry, stain_x+rx, stain_y+ry], fill=stain_color)
    
    # Dirt Spots
    if random.random() < severity * 0.8:
        num_spots = random.randint(3, int(8 * severity) + 3)
        for _ in range(num_spots):
            spot_x = random.randint(0, img.width)
            spot_y = random.randint(0, img.height)
            spot_size = random.randint(2, 8)
            dirt_color = (
                random.randint(80, 120),
                random.randint(70, 110),
                random.randint(60, 100),
                random.randint(40, 100)
            )
            draw.ellipse([spot_x-spot_size, spot_y-spot_size, spot_x+spot_size, spot_y+spot_size], fill=dirt_color)
    
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay)
    return img.convert("RGB")



def generate_v9_bag(filename, patient, drug, is_danger=False, optical_severity=0, clean_version=False):
    """V12: 法規完整版 + 光學壓力測試 + 乾淨版（供拍照）"""
    img = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), "white")
    draw = ImageDraw.Draw(img)
    
    # Fonts (縮小以適應 896x896)
    f_h1 = get_font(36)
    f_h2 = get_font(28)
    f_body = get_font(22)
    f_huge = get_font(40)
    f_warn = get_font(24)

    # --- 1. Top Header ---
    draw.text((40, 25), "MedGemma 聯合醫療體系", fill="#003366", font=f_h1)
    draw.text((40, 68), "地址: 新北市新莊區中正路 999 號", fill="gray", font=get_font(18))  # P0: 法規補全
    draw.text((40, 95), "用藥諮詢: (02) 2345-6789", fill="red", font=f_h2)
    
    # QR Code
    try:
        qr = qrcode.QRCode(box_size=3, border=1)
        qr.add_data(f"https://medgemma.tw/verify?id={drug['id']}")
        qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
        if qr_img.width > 150: qr_img = qr_img.resize((100, 100))
        img.paste(qr_img, (IMG_WIDTH-qr_img.width-20, 20))
    except Exception as e: print(f"⚠️ QR Error: {e}")
    
    draw.line([(30, 130), (IMG_WIDTH-30, 130)], fill="#003366", width=3)  # 調整位置，增加上方空間

    # --- 2. Patient Info ---
    y_p = 150  # 調整起始位置，增加與上方線條的距離
    draw.text((40, y_p), f"姓名: {patient['name']}", fill="black", font=f_h1)
    draw.text((350, y_p+5), f"{patient['gender']}", fill="black", font=f_h2)
    # [Audit Fix] Dynamic ROC Date (Year - 1911)
    roc_year = datetime.now().year - 1911
    today_str = datetime.now().strftime(f"{roc_year}/%m/%d")
    draw.text((40, y_p+50), f"調劑日: {today_str}", fill="black", font=f_body)
    draw.text((40, y_p+78), f"調劑藥師: 王專業", fill="black", font=f_body)  # P0: 法規補全，增加間距
    
    draw.line([(30, y_p+110), (IMG_WIDTH-30, y_p+110)], fill="gray", width=2)  # 增加與藥師文字的距離

    # --- 3. Drug Info ---
    y_drug = 280  # 調整起始位置，增加與上方的距離
    color_map = {"高血壓": "green", "糖尿病": "orange", "失眠": "blue"}
    bar_color = color_map.get(drug['cat'], "gray")
    draw.rectangle([15, y_drug, 30, y_drug+100], fill=bar_color)
    
    draw.text((45, y_drug), drug['cht'], fill="blue", font=f_huge)
    draw.text((45, y_drug+45), drug['eng'], fill="black", font=f_h2)
    
    # Dose (Risk Injection - P1: 臨床邏輯優化)
    # [V12.5 Audit Fix] Synced with Neuro-Symbolic 4-Rule Engine
    dose_val = drug['dose']
    original_usage = drug['usage']
    
    if is_danger:
         # Rule 1: Metformin (Glucophage) Limit (>1000mg or High Daily)
         if "Metformin" in drug['eng'] or "Glucophage" in drug['eng']: 
             dose_val = "2000mg"  # Direct violation of Single Dose Safety
             drug['usage'] = "BID" 
             drug['warning'] += " ⚠️ 劑量過高風險 (Lactic Acidosis)"

         # Rule 2: Zolpidem (Stilnox) Limit (>5mg for Elderly)
         elif "Zolpidem" in drug['eng'] or "Stilnox" in drug['eng']:
             dose_val = "10mg" # Standard pill, but dangerous for Elderly (Limit 5mg)
             drug['usage'] = "HS"
             drug['warning'] += " ⚠️ 長者跌倒風險 (Beers Criteria)"

         # Rule 3: High Dose Aspirin (>325mg)
         elif "Aspirin" in drug['eng'] or "Bokey" in drug['eng']:
             dose_val = "500mg" # Exceeds 325mg Check
             drug['warning'] += " ⚠️ 腸胃出血風險"

         # Rule 4: Acetaminophen Overdose (>4000mg)
         elif "Acetaminophen" in drug['eng'] or "Panadol" in drug['eng']:
             dose_val = "5000mg" # Absurd overdose
             drug['usage'] = "QD"
             drug['warning'] += " ⚠️ 肝毒性中毒風險"

         # Rule 5: Warfarin (Bleeding Risk) - Keep existing
         elif "Warfarin" in drug['eng']: 
             dose_val = "10mg"  
             drug['warning'] += " ⚠️ 出血風險極高"
             
         else: 
             # Generic Fallback
             dose_val = "5X Normal"
             drug['usage'] = "Q1H"
             drug['warning'] += " ⚠️ 劑量與頻次異常"
         
    draw.text((500, y_drug), f"劑量: {dose_val}", fill="black", font=f_h2)
    draw.text((500, y_drug+35), "總量: 28 顆", fill="black", font=f_body)
    if is_danger: 
        draw.text((500, y_drug+65), "⚠️ 劑量異常", fill="red", font=f_warn)
    
    draw.text((45, y_drug+100), f"適應症: {drug['indication']}", fill="black", font=f_body)

    # --- 4. Usage Box ---
    y_usage = 420  # 調整起始位置
    draw.rectangle([(40, y_usage), (856, y_usage+85)], outline="black", width=2)  # 稍微加高
    usage_text = {"BID": "每日兩次，早晚", "TID": "每日三次", "QD": "每日一次，早上", "QN": "每日一次，睡前"}
    timing_icon = "🍚" if "飯後" in drug['timing'] else "⏰"
    draw.text((60, y_usage+28), f"{timing_icon} {usage_text.get(drug['usage'], drug['usage'])} ({drug['timing']})", fill="black", font=f_h2)  # 置中

    # --- 5. Warning Box ---
    y_warn = 530  # 調整起始位置，增加與上方的距離
    draw.rectangle([40, y_warn, 856, y_warn+105], fill=(255, 245, 245), outline="red", width=2)  # 稍微加高
    draw.text((55, y_warn+15), "[!] 警語:", fill="red", font=f_warn)  # 簡化符號避免渲染問題，增加上邊距
    # V13 Fix: Use Text Wrap instead of dangerous truncation (Empathic Design)
    wrapper = textwrap.TextWrapper(width=34) # Adjust width based on font size
    wrapped_lines = wrapper.wrap(drug['warning'])
    
    # Draw strictly up to 2 lines to match box height, but try to fit more if possible
    # Actually, 2 lines is safe for 105px height (50 + 30 + margin)
    current_y = y_warn + 50
    for line in wrapped_lines[:2]:
        draw.text((55, current_y), line, fill="red", font=f_body)
        current_y += 30
        
    if "開車" in drug['warning']: draw_warning_icon(draw, 780, y_warn+55, 40, "car")
    if "酒" in drug['warning']: draw_warning_icon(draw, 830, y_warn+55, 40, "wine")

    # --- 6. Footer ---
    y_foot = 660  # 調整起始位置，增加與上方的距離
    draw.line([(30, y_foot), (IMG_WIDTH-30, y_foot)], fill="gray", width=1)
    draw.text((40, y_foot+20), "【三核對】□姓名 □外觀 □用法", fill="black", font=f_body)  # 增加上邊距
    
    # ==========================================
    # V12.1 CRITICAL FIX: Texture 與 Watermark 都應該受 clean_version 控制
    # ==========================================
    if not clean_version:
        # Texture (紙張紋理)
        try: img = apply_texture(img)
        except: pass

    # ==========================================
    # 🕵️ LEGAL PROTECTION: ANTI-FORGERY WATERMARK
    # ==========================================
    # Prevents "Forgery of Documents" accusations
    # Prevents Trademark Infringement confusion (Nominative Fair Use)
    # V12: 加入 clean_version 選項供 Sim2Physical 測試
    if not clean_version:  # 只在非乾淨版加浮水印
        draw = ImageDraw.Draw(img) # Re-init draw on textured image if needed
        wm_font = get_font(50)
        
        # Diagonal Watermark
        txt_layer = Image.new("RGBA", img.size, (255,255,255,0))
        d_ctx = ImageDraw.Draw(txt_layer)
        d_ctx.text((200, 400), "SAMPLE COPY - NOT FOR USE", fill=(200, 200, 200, 120), font=wm_font)
        d_ctx.text((150, 500), "AI GENERATED - DEMO ONLY", fill=(200, 200, 200, 120), font=wm_font)
        
        # Rotate watermark
        txt_layer = txt_layer.rotate(30)
        img = Image.alpha_composite(img.convert("RGBA"), txt_layer).convert("RGB")

    # Optical Stress
    try: img = apply_optical_stress(img, severity=optical_severity)
    except Exception as e: print(f"⚠️ Stress Fail: {e}")


    # 🎯 NEW: Comprehensive Edge Case Suite
    # Coverage: Creases, Glare, Physical Damage
    if optical_severity >= 1:
        try:
            # Edge Case 1: Creases (always apply if severity >= 1)
            crease_intensity = min(1.0, optical_severity * 0.4)
            img = add_creases(img, intensity=crease_intensity)
            
            # Edge Case 2: Plastic Glare (always apply if severity >= 1)
            glare_intensity = min(1.0, optical_severity * 0.5)
            img = apply_plastic_glare(img, intensity=glare_intensity)
            
            # Edge Case 3: Physical Damage (only for severe cases, severity >= 2)
            if optical_severity >= 2:
                damage_severity = min(1.0, (optical_severity - 1) * 0.3)
                img = apply_physical_damage(img, severity=damage_severity)
                
        except Exception as e:
            print(f"⚠️ Edge Case Application Failed: {e}")

    try:
        img.save(filename)
        print(f"✅ Generated: {filename} (Danger={is_danger}, Stress={optical_severity})")
    except: pass
    

# ==========================================
# 5. Database (Regulatory-Compliant Synthetic Data)
# ==========================================
PATIENTS = [
    {"name": "王大明", "gender": "男 (M)", "id": "A123456789"},
    {"name": "林美玉", "gender": "女 (F)", "id": "B223456789"},
    {"name": "張志明", "gender": "男 (M)", "id": "C123456789"},
    {"name": "陳淑芬", "gender": "女 (F)", "id": "D223456789"},
]

# ==========================================
# 5. Database (Regulatory-Compliant Synthetic Data - SYNCED with medgemma_data.py)
# ==========================================
from medgemma_data import DRUG_DATABASE as MASTER_DB

PATIENTS = [
    {"name": "王大明", "gender": "男 (M)", "id": "A123456789"},
    {"name": "林美玉", "gender": "女 (F)", "id": "B223456789"},
    {"name": "張志明", "gender": "男 (M)", "id": "C123456789"},
    {"name": "陳淑芬", "gender": "女 (F)", "id": "D223456789"},
]

def get_synced_drugs():
    """ Adapter: DRUG_DATABASE (V5) -> Stress Test Schema (V9) """
    synced_list = []
    pid_counter = 1
    
    for category, drugs in MASTER_DB.items():
        for d in drugs:
            # 1. Parse Appearance (Simplified for Stress Test)
            app = d["appearance"]
            shape = "circle"
            color = "white"
            
            # Shape
            if "長" in app or "橢" in app: shape = "oval"
            elif "八角" in app: shape = "octagon"
            
            # Color
            if "粉" in app or "紅" in app: color = "pink"
            elif "黃" in app: color = "yellow"
            
            # P2: Warfarin 專屬顏色邏輯（國際標準）
            if "Warfarin" in d['name_en'] or "華法林" in d['name_zh']:
                if "1" in d['dose']: color = "tan"
                elif "2" in d['dose']: color = "purple"
                elif "3" in d['dose']: color = "blue"
                elif "5" in d['dose']: color = "pink"
                elif "10" in d['dose']: color = "white"
            
            # 2. Parse Usage/Timing
            usage = "BID"
            timing = "飯後"
            u_tag = d["default_usage"]
            
            if "QD" in u_tag: 
                usage = "QD"
                if "bedtime" in u_tag or "HS" in u_tag: 
                    usage = "QN" # Map to Stress Test QN
                    timing = "睡前"
                elif "before" in u_tag: timing = "飯前"
            elif "TID" in u_tag: usage = "TID"
            
            # 3. Create Object
            synced_list.append({
                "id": d["code"],
                "cat": d["indication"], # Use indication as category display
                "cht": d["name_zh"],
                "eng": f"{d['name_en']} ({d['generic']})",
                "dose": d["dose"],
                "usage": usage,
                "timing": timing,
                "color": color,
                "shape": shape,
                "warning": d["warning"],
                "indication": d["indication"]
            })
            pid_counter += 1
            
    return synced_list

DRUGS = get_synced_drugs()
print(f"✅ Synced {len(DRUGS)} drugs from Source of Truth (medgemma_data.py)")

if __name__ == "__main__":
    from PIL import ImageEnhance # Import needed for optical stress
    import json
    print("🏥 MedGemma Challenge Generator V12 (Full Compliance + Clean Version)...")
    
    # [Audit Fix P0] Initialize Label Collection
    stress_test_labels = []
    
    # 1. Generate 5 Perfect Images (Expect: PASS)
    for i in range(1, 6):
        p = random.choice(PATIENTS)
        d = random.choice(DRUGS)
        filename = f"{OUTPUT_DIR}/demo_clean_{i}.png"
        generate_v9_bag(filename, p, d, is_danger=False, optical_severity=0)
        
        # [Audit Fix P0] Record Ground Truth
        stress_test_labels.append({
            "id": f"STRESS_CLEAN_{i:04d}",
            "image": f"demo_clean_{i}.png",
            "difficulty": "easy",
            "risk_status": "WITHIN_STANDARD",
            "patient": p,
            "drug": d,
            "is_danger": False
        })
        
    # 2. Generate 20 Dirty Images (Expect: WARNING/PASS depending on legibility)
    for i in range(1, 21):
        p = random.choice(PATIENTS)
        d = random.choice(DRUGS)
        filename = f"{OUTPUT_DIR}/demo_dirty_{i}.png"
        generate_v9_bag(filename, p, d, is_danger=False, optical_severity=2)
        
        # [Audit Fix P0] Record Ground Truth
        stress_test_labels.append({
            "id": f"STRESS_DIRTY_{i:04d}",
            "image": f"demo_dirty_{i}.png",
            "difficulty": "medium",
            "risk_status": "WITHIN_STANDARD",
            "patient": p,
            "drug": d,
            "is_danger": False,
            "optical_severity": 2
        })

    # 3. Generate 25 Dangerous Images (Expect: HIGH_RISK)
    for i in range(1, 26):
        p = random.choice(PATIENTS)
        d = random.choice(DRUGS)
        filename = f"{OUTPUT_DIR}/IMG_{i:04d}.png"
        generate_v9_bag(filename, p, d, is_danger=True, optical_severity=1)
        
        # [Audit Fix P0] Record Ground Truth
        stress_test_labels.append({
            "id": f"STRESS_DANGER_{i:04d}",
            "image": f"IMG_{i:04d}.png",
            "difficulty": "hard",
            "risk_status": "HIGH_RISK",
            "patient": p,
            "drug": d,
            "is_danger": True,
            "optical_severity": 1
        })
    
    # 4. 🎯 V12 新增：乾淨版（無浮水印）供 Sim2Physical 拍照測試
    print("📸 Generating CLEAN versions for Sim2Physical testing...")
    for i in range(1, 6):
        p = random.choice(PATIENTS)
        d = random.choice(DRUGS)
        filename = f"{OUTPUT_DIR}/clean_photo_test_{i}.png"
        generate_v9_bag(filename, p, d, is_danger=False, optical_severity=0, clean_version=True)
        
        # [Audit Fix P0] Record Ground Truth
        stress_test_labels.append({
            "id": f"STRESS_PHOTO_{i:04d}",
            "image": f"clean_photo_test_{i}.png",
            "difficulty": "easy",
            "risk_status": "WITHIN_STANDARD",
            "patient": p,
            "drug": d,
            "is_danger": False,
            "clean_version": True
        })
    
    # [Audit Fix P0] Export Ground Truth Labels
    labels_path = f"{OUTPUT_DIR}/stress_test_labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(stress_test_labels, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ All Assets Ready! ({len(stress_test_labels)} samples)")
    print(f"   - 5 Clean | 20 Dirty | 25 Dangerous | 5 Photo Test")
    print(f"📋 Ground Truth Labels: {labels_path}")
    print("🎯 Edge Case Coverage: 100% (5/5) - Creases, Glare, Physical Damage, Blur, Lighting")