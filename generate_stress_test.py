"""
Gallery of Horrors - Stress Test Image Generator (Taiwan Standard Edition)
========================================================================
Generates 10 "nightmare" prescription images that strictly adhere to 
Taiwan "Pharmacist Act" (藥師法) labeling regulations.

Features:
- Standard "門診藥袋" Layout
- Full Chinese Fields (姓名, 病歷號, 調劑日期, 適應症)
- QR Code (Smart Hospital Feature)
- Drug Appearance Description
- Noto Sans CJK TC Font (Professional Printing Style)

Usage:
    python generate_stress_test.py
"""

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import os
import random
import json
import requests
import qrcode
import numpy as np
from datetime import datetime, timedelta

# Create output directory
OUTPUT_DIR = "assets/stress_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Image size matches Training Data
IMG_SIZE = 896

# ===== 1. Data & Config (Synced from KAGGLE_V4) =====

HOSPITAL_INFO = {
    "name": "MedGemma 智慧醫療示範醫院",
    "address": "台北市信義區信義路五段7號",
    "phone": "(02) 8765-4321",
    "pharmacist": "王大明",
    "checker": "李小美"
}

# Simplified Drug DB for Stress Test
DRUG_DB = [
    {"name_en": "Glucophage", "name_zh": "庫魯化", "generic": "Metformin", "dose": "500mg", "appearance": "白色長圓形", "indication": "降血糖", "warning": "隨餐服用", "usage": {"text_zh": "每日兩次 早晚飯後", "quantity": 56}},
    {"name_en": "Norvasc", "name_zh": "脈優", "generic": "Amlodipine", "dose": "5mg", "appearance": "白色八角形", "indication": "降血壓", "warning": "小心姿勢性低血壓", "usage": {"text_zh": "每日一次 早餐飯後", "quantity": 28}},
    {"name_en": "Stilnox", "name_zh": "使蒂諾斯", "generic": "Zolpidem", "dose": "10mg", "appearance": "白色長條形", "indication": "失眠", "warning": "服用後請立即就寢", "usage": {"text_zh": "每日一次 睡前服用", "quantity": 28}},
    {"name_en": "Aspirin", "name_zh": "阿斯匹靈", "generic": "ASA", "dose": "100mg", "appearance": "白色圓形", "indication": "預防血栓", "warning": "胃潰瘍患者慎用", "usage": {"text_zh": "每日一次 早餐飯後", "quantity": 28}},
    {"name_en": "Lipitor", "name_zh": "立普妥", "generic": "Atorvastatin", "dose": "20mg", "appearance": "白色橢圓形", "indication": "降血脂", "warning": "肌肉痠痛時需回診", "usage": {"text_zh": "每日一次 睡前服用", "quantity": 28}},
]

# ===== 2. Font Logic =====
def download_font(font_name, url):
    if not os.path.exists(font_name):
        print(f"📥 Downloading font: {font_name}...")
        try:
            response = requests.get(url, timeout=30)
            with open(font_name, 'wb') as f:
                f.write(response.content)
        except Exception as e:
            print(f"⚠️ Font download failed: {e}")
    return font_name

def get_font_paths():
    # Priority 1: System
    sys_bold = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    sys_reg = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if os.path.exists(sys_bold): return sys_bold, sys_reg

    # Priority 2: Local Download
    bold_url = "https://raw.githubusercontent.com/googlefonts/noto-cjk/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Bold.otf"
    reg_url = "https://raw.githubusercontent.com/googlefonts/noto-cjk/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf"
    return download_font("NotoSansTC-Bold.otf", bold_url), download_font("NotoSansTC-Regular.otf", reg_url)

# ===== 3. Generator (Drawing) =====
def generate_base_prescription(drug_idx):
    drug = DRUG_DB[drug_idx % len(DRUG_DB)]
    
    # Generate random patient data
    date_str = (datetime.now() - timedelta(days=random.randint(0, 30))).strftime("%Y/%m/%d")
    chart_no = f"A{random.randint(100000, 999999)}"
    rx_id = f"R{random.randint(202600000000, 202699999999)}"
    
    img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), 'white')
    draw = ImageDraw.Draw(img)
    
    # Load fonts
    font_bold_path, font_reg_path = get_font_paths()
    try:
        ft_title = ImageFont.truetype(font_bold_path, 40)
        ft_large = ImageFont.truetype(font_bold_path, 36)
        ft_main = ImageFont.truetype(font_reg_path, 28) # Slightly larger for readability
        ft_small = ImageFont.truetype(font_reg_path, 24)
        ft_warn = ImageFont.truetype(font_bold_path, 24)
    except:
        ft_title = ImageFont.load_default()
        ft_large = ImageFont.load_default()
        ft_main = ImageFont.load_default()
        ft_small = ImageFont.load_default()
        ft_warn = ImageFont.load_default()

    # --- Header ---
    draw.text((40, 30), HOSPITAL_INFO["name"], font=ft_title, fill="#003366")
    draw.text((560, 80), "門診藥袋", font=ft_title, fill="black") # Standard Title (Moved Down)
    
    # QR Code (Smart Hospital)
    qr = qrcode.make(json.dumps({"id": rx_id, "drug": drug["name_en"]})).resize((110, 110))
    img.paste(qr, (740, 20))
    
    draw.line([(30, 140), (866, 140)], fill="#003366", width=4)
    
    # --- Patient Info ---
    # Row 1
    draw.text((50, 160), "姓名: 吳振明", font=ft_large, fill="black")
    draw.text((450, 165), f"病歷號: {chart_no}", font=ft_main, fill="black")
    
    # Row 2
    draw.text((50, 210), "年齡: 78 歲", font=ft_large, fill="black")
    draw.text((450, 215), f"調劑日: {date_str}", font=ft_main, fill="black")
    
    draw.line([(30, 270), (866, 270)], fill="gray", width=2)
    
    # --- Drug Info ---
    # English Name + Dose
    draw.text((50, 290), f"{drug['name_en']} {drug['dose']}", font=ft_title, fill="black")
    # Chinese Name + Generic
    draw.text((50, 340), f"{drug['name_zh']} ({drug['generic']})", font=ft_main, fill="#444444")
    # Quantity
    draw.text((600, 290), f"總量: {drug['usage']['quantity']}", font=ft_large, fill="black")
    
    # Appearance (New Field)
    draw.text((50, 390), f"外觀: {drug['appearance']}", font=ft_main, fill="#006600") # Dark Green
    
    # --- Usage Box ---
    draw.rectangle([(40, 440), (850, 540)], outline="black", width=3)
    draw.text((60, 470), drug['usage']['text_zh'], font=ft_title, fill="black")
    
    # --- Indication & Warning ---
    y_base = 580
    draw.text((50, y_base), "適應症:", font=ft_main, fill="black")
    draw.text((160, y_base), drug['indication'], font=ft_main, fill="black")
    
    draw.text((50, y_base+50), "⚠ 警語:", font=ft_warn, fill="red")
    draw.text((160, y_base+50), drug['warning'], font=ft_main, fill="red")
    
    # Footer
    draw.line([(30, 800), (866, 800)], fill="gray", width=1)
    draw.text((50, 820), f"藥師: {HOSPITAL_INFO['pharmacist']}  核對: {HOSPITAL_INFO['checker']}", font=ft_small, fill="gray")
    draw.text((50, 850), f"地址: {HOSPITAL_INFO['address']}  電話: {HOSPITAL_INFO['phone']}", font=ft_small, fill="gray")

    return img

# ===== 4. Distortions (The Gallery of Horrors) =====
def apply_extreme_blur(img, intensity="heavy"):
    if intensity == "heavy": return img.filter(ImageFilter.GaussianBlur(radius=8))
    elif intensity == "motion": return img.filter(ImageFilter.BoxBlur(radius=6))
    return img

def apply_low_light(img):
    enhancer = ImageEnhance.Brightness(img); img = enhancer.enhance(0.4)
    enhancer = ImageEnhance.Contrast(img); return enhancer.enhance(0.8)

def apply_overexposure(img):
    enhancer = ImageEnhance.Brightness(img); img = enhancer.enhance(1.8)
    enhancer = ImageEnhance.Contrast(img); return enhancer.enhance(0.5)

def apply_noise(img, amount=50):
    arr = np.array(img)
    noise = np.random.randint(-amount, amount, arr.shape, dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def apply_crease(img):
    draw = ImageDraw.Draw(img)
    width, height = img.size
    draw.line([(0, height//3), (width, height//2)], fill=(180, 180, 180), width=10)
    return img

def apply_water_damage(img):
    draw = ImageDraw.Draw(img, 'RGBA')
    width, height = img.size
    for _ in range(3):
        x, y = random.randint(0, width), random.randint(0, height//2)
        r = random.randint(60, 150)
        draw.ellipse([(x-r, y-r), (x+r, y+r)], fill=(220, 210, 190, 80))
    return img.convert('RGB')

def apply_paper_texture(img):
    """
    Simulates crumpled paper texture using procedural noise overlay.
    Strategy: Generates a grayscale noise layer, blurs it to create 'folds', 
    and blends it with the original image using 'multiply' mode logic.
    """
    width, height = img.size
    # 1. Generate base noise map
    arr = np.random.randint(200, 255, (height, width), dtype=np.uint8)
    texture = Image.fromarray(arr, mode='L')
    
    # 2. Create 'folds' by blurring large noise blobs
    # (Simulating shadows of wrinkles)
    fold_map = np.random.randint(100, 220, (height // 4, width // 4), dtype=np.uint8)
    fold_img = Image.fromarray(fold_map, mode='L').resize((width, height), resample=Image.BICUBIC)
    fold_img = fold_img.filter(ImageFilter.GaussianBlur(radius=15))
    
    # 3. Blend logic (simulating Apply mode)
    # Convert original to RGBA to allow blending
    img = img.convert("RGBA")
    
    # Create overlay
    overlay = Image.new('RGBA', img.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    
    # In manual pixel manipulation or complex blending, we'd multiply.
    # Here, we'll just alpha blend the fold map as a shadow layer
    # Simply put: Darker folds
    
    # Convert fold_img to valid mask/layer
    # Using PIL's math to multiply intensity would be best but simple alpha blend is robust here
    # Let's use image composition:
    
    # Create a texture layer:
    # Use the fold image to darken the base image
    # We can multiply the arrays
    
    img_arr = np.array(img).astype(float)
    fold_arr = np.array(fold_img).astype(float) / 255.0
    
    # Expand dims for broadcasting (H, W, 1) if using just fold_arr
    fold_arr = np.expand_dims(fold_arr, axis=2)
    
    # Multiply: Darkens grid lines based on folds
    # Keep alpha channel intact if present, or just operate on RGB
    img_rgb = img_arr[:,:,:3] * fold_arr
    
    # Re-combine
    res_arr = np.dstack((img_rgb, img_arr[:,:,3])).astype(np.uint8)
    return Image.fromarray(res_arr).convert("RGB")

def apply_skew(img, angle=15):
    return img.rotate(angle, expand=True, fillcolor="white")

def apply_occlusion(img):
    draw = ImageDraw.Draw(img)
    width, height = img.size
    # Occlude drug name partially
    draw.ellipse([(100, 280), (300, 350)], fill=(210, 180, 160)) # Finger-like tone
    return img

STRESS_TESTS = [
    ("01_extreme_blur", "Extreme Gaussian Blur", lambda img: apply_extreme_blur(img, "heavy")),
    ("02_motion_blur", "Motion Blur", lambda img: apply_extreme_blur(img, "motion")),
    ("03_low_light", "Dark/Low Light", apply_low_light),
    ("04_overexposed", "Overexposed", apply_overexposure),
    ("05_heavy_noise", "Heavy Noise", lambda img: apply_noise(img, 80)),
    ("06_paper_texture", "Paper Texture (Physical Augmentation)", apply_paper_texture),
    ("07_water_damage", "Water Damage", apply_water_damage),
    ("08_skewed_angle", "Skewed 25°", lambda img: apply_skew(img, 25)),
    ("09_occlusion", "Finger Occlusion", apply_occlusion),
    ("10_combined_hell", "Combined (Blur+Noise+Texture)", lambda img: apply_noise(apply_low_light(apply_paper_texture(img)), 40)),
]

if __name__ == "__main__":
    print("🎭 Generating 'Gallery of Horrors' (Taiwan Standard Edition)...")
    print("=" * 60)
    
    # Ensure fonts valid
    get_font_paths()
    
    for i, (filename, description, transform_fn) in enumerate(STRESS_TESTS):
        print(f"  👉 Generating {filename}...")
        base = generate_base_prescription(i)
        stressed = transform_fn(base)
        stressed.save(os.path.join(OUTPUT_DIR, f"{filename}.png"))
        
    print("=" * 60)
    print(f"🎉 Done! 10 Taiwan-Compliant Stress Test Images saved to {OUTPUT_DIR}/")
