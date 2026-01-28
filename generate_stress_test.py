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
# V10 FIX: 改為 896x896 與訓練資料一致
IMG_WIDTH = 896
IMG_HEIGHT = 896

# ==========================================
# 1. 資源準備 (Auto-Font)
# ==========================================
FONT_URL = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf"
FONT_PATH = "NotoSansCJKtc-Regular.otf"

def get_font(size):
    """取得字型，帶完整錯誤處理避免 Kaggle 崩潰"""
    if not os.path.exists(FONT_PATH):
        try:
            print(f"⬇️ 下載中文字體中... ({size}px)")
            r = requests.get(FONT_URL, timeout=30)
            r.raise_for_status()  # 檢查 HTTP 錯誤
            with open(FONT_PATH, "wb") as f:
                f.write(r.content)
            print(f"   ✅ 字體下載成功")
        except Exception as e:
            pass # Keep silent fall back
    
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception as e:
        print(f"   ⚠️ 字體載入失敗: {e}，使用預設字體")
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
# New V11 Feature: Optical Corruption Module
# ==========================================
def apply_optical_stress(img, severity=0):
    """
    Simulate real-world challenging conditions.
    severity: 0 (None), 1 (Mild - Hand tremor), 2 (Hard - Bad focus/lighting)
    """
    if severity == 0: return img
    
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
# I will modify generate_v9_bag to accept `optical_severity` and call `apply_optical_stress` at the end.

def generate_v9_bag(filename, patient, drug, is_danger=False, optical_severity=0):
    """V11: 896x896 版本，支援光學壓力測試"""
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
    draw.text((40, 70), "用藥諮詢: (02) 2345-6789", fill="red", font=f_h2)
    
    # QR Code
    try:
        qr = qrcode.QRCode(box_size=3, border=1)
        qr.add_data(f"https://medgemma.tw/verify?id={drug['id']}")
        qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
        if qr_img.width > 150: qr_img = qr_img.resize((100, 100))
        img.paste(qr_img, (IMG_WIDTH-qr_img.width-20, 20))
    except Exception as e: print(f"⚠️ QR Error: {e}")
    
    draw.line([(30, 110), (IMG_WIDTH-30, 110)], fill="#003366", width=3)

    # --- 2. Patient Info ---
    y_p = 130
    draw.text((40, y_p), f"姓名: {patient['name']}", fill="black", font=f_h1)
    draw.text((350, y_p+5), f"{patient['gender']}", fill="black", font=f_h2)
    draw.text((40, y_p+45), f"調劑日: 115/01/22", fill="black", font=f_body)
    
    draw.line([(30, y_p+80), (IMG_WIDTH-30, y_p+80)], fill="gray", width=2)

    # --- 3. Drug Info ---
    y_drug = 230
    color_map = {"高血壓": "green", "糖尿病": "orange", "失眠": "blue"}
    bar_color = color_map.get(drug['cat'], "gray")
    draw.rectangle([15, y_drug, 30, y_drug+100], fill=bar_color)
    
    draw.text((45, y_drug), drug['cht'], fill="blue", font=f_huge)
    draw.text((45, y_drug+45), drug['eng'], fill="black", font=f_h2)
    
    # Dose (Risk Injection)
    dose_val = drug['dose']
    if is_danger:
         # Strategic Risk Injection: Not just 5000mg
         if "Metformin" in drug['eng']: dose_val = "2500mg (OD)" # Overdose
         elif "Warfarin" in drug['eng']: dose_val = "10mg" # High bleeding risk
         else: dose_val = "5000mg"
         
    draw.text((500, y_drug), f"劑量: {dose_val}", fill="black", font=f_h2)
    draw.text((500, y_drug+35), "總量: 28 顆", fill="black", font=f_body)
    if is_danger: 
        draw.text((500, y_drug+65), "⚠️ 劑量異常", fill="red", font=f_warn)
    
    draw.text((45, y_drug+100), f"適應症: {drug['indication']}", fill="black", font=f_body)

    # --- 4. Usage Box ---
    y_usage = 370
    draw.rectangle([(40, y_usage), (856, y_usage+80)], outline="black", width=2)
    usage_text = {"BID": "每日兩次，早晚", "TID": "每日三次", "QD": "每日一次，早上", "QN": "每日一次，睡前"}
    timing_icon = "🍚" if "飯後" in drug['timing'] else "⏰"
    draw.text((60, y_usage+25), f"{timing_icon} {usage_text.get(drug['usage'], drug['usage'])} ({drug['timing']})", fill="black", font=f_h2)

    # --- 5. Warning Box ---
    y_warn = 480
    draw.rectangle([40, y_warn, 856, y_warn+100], fill=(255, 245, 245), outline="red", width=2)
    draw.text((55, y_warn+10), "⚠️ 警語:", fill="red", font=f_warn)
    warning_text = drug['warning'][:30] + "..." if len(drug['warning']) > 30 else drug['warning']
    draw.text((55, y_warn+45), warning_text, fill="red", font=f_body)
    if "開車" in drug['warning']: draw_warning_icon(draw, 780, y_warn+50, 40, "car")
    if "酒" in drug['warning']: draw_warning_icon(draw, 830, y_warn+50, 40, "wine")

    # --- 6. Footer ---
    y_foot = 610
    draw.line([(30, y_foot), (IMG_WIDTH-30, y_foot)], fill="gray", width=1)
    draw.text((40, y_foot+15), "【三核對】□姓名 □外觀 □用法", fill="black", font=f_body)
    
    # Texture
    try: img = apply_texture(img)
    except: pass

    # ==========================================
    # 🕵️ LEGAL PROTECTION: ANTI-FORGERY WATERMARK
    # ==========================================
    # Prevents "Forgery of Documents" accusations
    # Prevents Trademark Infringement confusion (Nominative Fair Use)
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

    try:
        img.save(filename)
        print(f"✅ Generated: {filename} (Danger={is_danger}, Stress={optical_severity})")
    except: pass

# ==========================================
# 5. Database (Authentic Taiwan Data)
# ==========================================
PATIENTS = [
    {"name": "王大明", "gender": "男 (M)", "id": "A123456789"},
    {"name": "林美玉", "gender": "女 (F)", "id": "B223456789"},
    {"name": "張志明", "gender": "男 (M)", "id": "C123456789"},
    {"name": "陳淑芬", "gender": "女 (F)", "id": "D223456789"},
]

# ==========================================
# 5. Database (Authentic Taiwan Data - SYNCED)
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
    print("🏥 MedGemma Challenge Generator V11 (Strategic + Optical Stress)...")
    
    # 1. Generate 3 Perfect Images (Expect: PASS)
    for i in range(1, 4):
        p = random.choice(PATIENTS)
        d = random.choice(DRUGS)
        generate_v9_bag(f"{OUTPUT_DIR}/demo_clean_{i}.jpg", p, d, is_danger=False, optical_severity=0)
        
    # 2. Generate 1 High Risk Image (Expect: HIGH_RISK logic trap)
    p = PATIENTS[0] # High age
    d = DRUGS[0] # Metformin
    generate_v9_bag(f"{OUTPUT_DIR}/demo_high_risk.jpg", p, d, is_danger=True, optical_severity=0)
    
    # 3. Generate 1 Bad Quality Image (Expect: REJECT / INPUT GATE TRIGGER)
    p = random.choice(PATIENTS)
    d = random.choice(DRUGS)
    generate_v9_bag(f"{OUTPUT_DIR}/demo_blur_reject.jpg", p, d, is_danger=False, optical_severity=2)
    
    print("🚀 All Challenge Assets Ready!")