import os
import random
import math
import requests
import json
import datetime
import qrcode
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps

# [V26] Data Sync Import
try:
    import medgemma_data
    DATA_SYNC_AVAILABLE = True
except ImportError:
    print("⚠️ medgemma_data.py not found. Using local fallback.")
    DATA_SYNC_AVAILABLE = False


# ==========================================
# ⚖️ LEGAL DISCLAIMER / 免責聲明
# ==========================================
# This script generates SYNTHETIC medical data for AI research and training purposes only.
# The generated images do NOT represent real patient data, medical advice, or actual drug prescriptions.
# All names, IDs, and QR code data are fictional or anonymized.
# 
# 本程式僅供 AI 研究與演算法訓練使用，產出之圖像均為合成數據。
# 內容不含真實病患資料，亦不代表真實醫療建議。
# ==========================================

# ==========================================
# 1. 基礎配置 (Setup): V25 Safety & Compliance
# ==========================================
OUTPUT_DIR = "assets/lasa_dataset_v17_compliance"
os.makedirs(OUTPUT_DIR, exist_ok=True)
IMG_SIZE = 896

# Google Fonts URLs
FONT_URLS = {
    "Bold": "https://github.com/google/fonts/raw/main/ofl/notosanstc/NotoSansTC-Bold.otf",
    "Regular": "https://github.com/google/fonts/raw/main/ofl/notosanstc/NotoSansTC-Regular.otf"
}
FONT_PATHS = {
    "Bold": "NotoSansTC-Bold.otf",
    "Regular": "NotoSansTC-Regular.otf"
}

def download_fonts():
    """Auto-download fonts. STRICT MODE: Fail if download fails."""
    # [V25] Tip: For Kaggle, upload these fonts to a Dataset and change FONT_PATHS to input path.
    for style, url in FONT_URLS.items():
        path = FONT_PATHS[style]
        if not os.path.exists(path):
            print(f"⬇️ Downloading font: {style}...")
            try:
                r = requests.get(url, timeout=15)
                if r.status_code == 200:
                    with open(path, "wb") as f:
                        f.write(r.content)
                    print(f"✅ Saved {path}")
                else:
                    raise Exception(f"HTTP {r.status_code}")
            except Exception as e:
                print(f"❌ CRITICAL: Font download failed for {style}. Text will be garbage.")
                raise RuntimeError(f"Font download failed: {e}")

def get_font(size, bold=False):
    path = FONT_PATHS["Bold"] if bold else FONT_PATHS["Regular"]
    try:
        return ImageFont.truetype(path, size)
    except:
        # Retry download once
        try:
             download_fonts()
             return ImageFont.truetype(path, size)
        except Exception as e:
             print(f"❌ FATAL ERROR: Cannot load font {path}.")
             raise RuntimeError("Font loading failed. Aborting to prevent data corruption.")

def get_jitter_offset(amp=5):
    """Return a fixed (dx, dy) to apply to a whole group of elements."""
    return random.randint(-amp, amp), random.randint(-amp, amp)

def fit_text_to_width(draw, text, font_path, max_width, start_size=50):
    """Recursively shrink font size until text fits in max_width."""
    size = start_size
    font = ImageFont.truetype(font_path, size)
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        width = bbox[2] - bbox[0]
    except AttributeError:
        # Fallback for older PIL
        width = draw.textlength(text, font=font)
    
    while width > max_width and size > 20:
        size -= 2
        font = ImageFont.truetype(font_path, size)
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            width = bbox[2] - bbox[0]
        except:
             width = draw.textlength(text, font=font)
        
    return font, size

# ==========================================
# 2. 雙重擬真引擎 V2 (Sim2Real with Specular)
# ==========================================

def add_creases(img, intensity=0.5):
    """[Layer 1: Physics] 物理摺痕"""
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for _ in range(random.randint(2, 5)):
        x1 = random.randint(0, img.width); y1 = random.randint(0, img.height)
        x2 = random.randint(0, img.width); y2 = random.randint(0, img.height)
        width = random.randint(1, 3)
        draw.line([(x1, y1), (x2, y2)], fill=(120, 120, 120, random.randint(30, 80)), width=width)
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay)
    return img.convert("RGB")

def apply_plastic_glare(img, intensity=0.5):
    """[Layer 3: Specular] 塑膠反光與材質模擬"""
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Simulating overhead tube lights reflection (Long white blobs)
    for _ in range(random.randint(1, 3)):
        w = random.randint(100, 400)
        h = random.randint(10, 40)
        x = random.randint(0, img.width - w)
        y = random.randint(0, img.height - h)
        
        # Soft white glare
        draw.ellipse([x, y, x+w, y+h], fill=(255, 255, 255, random.randint(20, 60)))
        
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay)
    return img.convert("RGB")

def apply_optical_stress(img, severity=0.5):
    """[Layer 2: Optics] 光學干擾"""
    if severity == 0: return img
    
    # 1. Blur (Defocus) - Adjusted by severity
    # severity 0.5 -> radius 1.0
    # severity 1.0 -> radius 2.0
    radius = severity * 2.0
    if radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(radius*0.8, radius*1.2)))
    
    # 2. Contrast/Brightness (Lighting conditions)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))
    
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.9, 1.1))
    
    return img

def draw_watermark(img):
    """[V25] Legal Compliance Watermark"""
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Faint repeated watermark
    text = "SYNTHETIC SAMPLE - NOT FOR MEDICAL USE"
    font = get_font(24, bold=True)
    
    # Diagonal placement
    cx, cy = img.width // 2, img.height // 2
    
    # Rotate text logic (simplified by drawing on rotated layer)
    # Actually, simplistic watermark: Bottom right corner and Top Left
    draw.text((20, 10), text, fill=(200, 200, 200, 120), font=font)
    draw.text((img.width - 500, img.height - 30), text, fill=(200, 200, 200, 120), font=font)
    
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay)
    return img.convert("RGB")

# ==========================================
# 3. 藥學知識庫 (LASA Dataset) [V26 Synced]
# ==========================================
if DATA_SYNC_AVAILABLE:
    print("🔄 [V26] Syncing with MedGemma Guardian Database...")
    LASA_PAIRS = medgemma_data.get_renderable_data()
else:
    # Fallback to local subset if sync fails
    LASA_PAIRS = {
        "SOUND_ALIKE_CRITICAL": [
            {"name": "Lasix 40mg (Furosemide)", "shape": "circle", "color": "white", "code": "ABC1234567", "indi": "高血壓/水腫", "zh": "來適泄錠", "warning": "警語: 服用後排尿頻繁，請避免睡前服用。\nSide Effect: Frequent urination.", "usage_code": "BID", "license": "衛署藥製字第012345號"},
            {"name": "Losec 20mg (Omeprazole)", "shape": "capsule", "color": "pink_brown", "code": "DEF8901234", "indi": "胃潰瘍/逆流性食道炎", "zh": "樂酸克膠囊", "warning": "警語: 飯前服用效果最佳，整顆吞服不可嚼碎。\nTake before meals. Swallow whole.", "usage_code": "QD", "license": "衛署藥輸字第067890號"}, 
        ],
        "LOOK_ALIKE_SHAPE": [
            {"name": "Xarelto 15mg", "shape": "circle", "color": "red", "code": "XAR1500001", "indi": "預防中風/血栓", "zh": "拜瑞妥膜衣錠", "warning": "警語: 隨餐服用。請注意出血徵兆。\nTake with food. Watch for bleeding.", "usage_code": "QD", "license": "衛部藥輸字第025888號"},
            {"name": "Dilatrend 25mg", "shape": "circle_scored", "color": "white_gold", "code": "DIL2500002", "indi": "高血壓/心衰竭", "zh": "達利全錠", "warning": "警語: 需長期服用，不可擅自停藥。\nDo not stop taking abruptly.", "usage_code": "BID", "license": "衛署藥輸字第011223號"},
        ]
    }

# ==========================================
# 4. 3D 藥丸渲染引擎 (Hyper-Real Pill)
# ==========================================
def draw_hyper_real_pill(draw, x, y, drug_data, force_mismatch=False):
    """
    Draw 3D-like pill.
    If force_mismatch is True, draw a DIFFERENT shape/color than defined in drug_data.
    """
    
    shape = drug_data['shape']
    color = drug_data['color']
    
    if force_mismatch:
        # Trap Mode: Swap shape/color to create hard negative
        if shape == "circle": shape = "capsule"; color = "pink_brown"
        elif shape == "capsule": shape = "circle"; color = "white"
        else: shape = "oblong"; color = "yellow"
        
    # Shadow
    draw.ellipse([x+5, y+55, x+85, y+75], fill=(200, 200, 200))
    
    # Body
    fill_color = {
        "white": "#F5F5F5", "yellow": "#FFF9C4", "pink": "#F8BBD0", "red": "#EF9A9A",
        "pink_brown": "#A1887F", "white_gold": "#FFF176", "brown_red": "#8D6E63", "peach": "#FFCCBC"
    }.get(color, "#E0E0E0")
    
    outline_color = "#616161"
    
    if shape == "circle":
        draw.ellipse([x, y, x+80, y+80], fill=fill_color, outline=outline_color, width=2)
        # Highlight
        draw.chord([x+10, y+10, x+70, y+70], start=135, end=225, fill=(255, 255, 255, 100))
        
    elif shape == "circle_scored":
        draw.ellipse([x, y, x+80, y+80], fill=fill_color, outline=outline_color, width=2)
        draw.line([x+40, y+10, x+40, y+70], fill=outline_color, width=2)
        
    elif shape == "capsule":
        draw.arc([x, y, x+80, y+40], start=180, end=0, fill=outline_color, width=2)
        draw.chord([x, y, x+80, y+80], start=0, end=180, fill=fill_color, outline=outline_color, width=2) # Bottom half
        draw.chord([x, y, x+80, y+80], start=180, end=360, fill=color if color!="pink_brown" else "#D7CCC8", outline=outline_color, width=2) # Top half
        
    elif shape == "oblong":
         draw.rounded_rectangle([x, y+20, x+100, y+60], radius=20, fill=fill_color, outline=outline_color, width=2)

# ==========================================
# 5. 格線系統 V17 (Grid System)
# ==========================================
def draw_bowl_full(draw, x, y, size):
    """飯後: 實心飯碗"""
    draw.chord([x-size, y, x+size, y+size], start=0, end=180, fill="black", outline="black")
    draw.line([x-size, y, x+size, y], fill="black", width=2)
    # Rice mound
    draw.chord([x-size+5, y-10, x+size-5, y], start=180, end=360, fill="black")

def draw_bowl_empty(draw, x, y, size):
    """飯前: 空碗"""
    draw.chord([x-size, y, x+size, y+size], start=0, end=180, fill=None, outline="black", width=2)
    draw.line([x-size, y, x+size, y], fill="black", width=2)

def draw_sun_rising(draw, x, y, size):
    """早上: 旭日 (Sun Rising)"""
    draw.arc([x-size, y, x+size, y+size*2], start=180, end=360, fill="black", width=2)
    draw.line([x-size-5, y+size, x+size+5, y+size], fill="black", width=2) # Horizon
    # Rays
    for i in range(3):
        angle = 180 + 45*(i+1)
        rad = math.radians(angle)
        sx = x + size * 1.2 * math.cos(rad)
        sy = y + size + size * 1.2 * math.sin(rad) 
        cx, cy = x, y+size
        sx = cx + size * 1.2 * math.cos(rad)
        sy = cy + size * 1.2 * math.sin(rad) 
        # Simple rays
        draw.line([x, y-size-5, x, y-size-15], fill="black", width=2) # Top
        draw.line([x-size, y-size/2, x-size-10, y-size/2-10], fill="black", width=2)
        draw.line([x+size, y-size/2, x+size+10, y-size/2-10], fill="black", width=2)

def draw_sun_full(draw, x, y, size):
    """中午: 烈日"""
    draw.ellipse([x-15, y-15, x+15, y+15], outline="black", width=2)
    # Rays around
    for i in range(0, 360, 45):
        rad = math.radians(i)
        x1 = x + 20 * math.cos(rad); y1 = y + 20 * math.sin(rad)
        x2 = x + 30 * math.cos(rad); y2 = y + 30 * math.sin(rad)
        draw.line([x1, y1, x2, y2], fill="black", width=2)

def draw_moon_evening(draw, x, y, size):
    """晚上: 彎月"""
    draw.chord([x-15, y-15, x+15, y+15], start=90, end=270, fill="black")
    draw.chord([x-10, y-15, x+10, y+15], start=90, end=270, fill="white") 
    draw.arc([x-15, y-15, x+15, y+15], start=60, end=300, fill="black", width=2)

def draw_moon_sleeping(draw, x, y, size):
    """睡前: 星月"""
    draw_moon_evening(draw, x-5, y, size)
    draw.text((x+10, y-10), "Zzz", fill="black", font=get_font(14))

# ==========================================
# 6. V25 Safety & Compliance (Privacy & Legal)
# ==========================================

# ==========================================
# 6. V26 Human Touch (Real Pharmacist & Dosage Info)
# ==========================================

HOSPITAL_META = {
    "name": "MedGemma 智慧示範藥局",
    "address": "台北市數位區虛擬路 0 號",
    "pharmacist": "王小明 (藥字第123456號)", # [V26] Human Pharmacist
    "phone": "02-0000-0000"
}

def mask_privacy_info(text, ratio=1.0):
    """
    [V25 Privacy] Enforce 100% masking for compliance.
    """
    # Always mask
    chars = list(text)
    if len(chars) <= 2: return text, False
    start = len(chars) // 3
    end = 2 * len(chars) // 3
    for i in range(start, end):
        chars[i] = "○" if "\u4e00" <= chars[i] <= "\u9fff" else "*"
    return "".join(chars), True

def get_pill_description(drug_data):
    shape_map = {
        "capsule": "膠囊", "circle": "圓形", "oval": "橢圓形", 
        "circle_scored": "圓形(刻痕)", "oblong": "長橢圓"
    }
    color_map = {
        "white": "白色", "yellow": "黃色", "pink": "粉紅", "red": "紅色",
        "pink_brown": "粉紅/紅棕", "white_gold": "白色/金色",
        "brown_red": "紅褐色", "peach": "桃紅色"
    }
    s = shape_map.get(drug_data['shape'], "多邊形")
    c = color_map.get(drug_data['color'], "雜色")
    return f"{c} {s}"

def generate_real_qr_code(data_dict, size=120):
    """
    Generate REAL QR Code with JSON data.
    Provides Digital Redundancy for VLM.
    """
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H, 
        box_size=10,
        border=1,
    )
    # [V25 Optimize] Minify JSON to save QR density
    minified = {}
    if "nhi" in data_dict: minified["id"] = data_dict["nhi"]
    if "name" in data_dict: minified["n"] = data_dict["name"]
    if "patient" in data_dict: minified["p"] = data_dict["patient"]
    if "indi" in data_dict: minified["i"] = data_dict["indi"]
    
    qr.add_data(json.dumps(minified, ensure_ascii=False))
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img = img.resize((size, size), Image.NEAREST)
    return img

def generate_v26_human_bag(filename, pair_type, drug_data, trap_mode=False, **kwargs):
    # 🎨 1. Base Layer
    # Use external params if provided, else random
    if "bg_color" in kwargs:
        bg_name = kwargs["bg_color"]
        # Map name to RGB
        bg_map = {
            "white": (255, 255, 255),
            "cream": (255, 253, 208), 
            "light_gray": (240, 240, 240),
            "warm_white": (255, 250, 240)
        }
        bg_color = bg_map.get(bg_name, (255, 255, 255))
    else:
        bg_color = (random.randint(245, 255), random.randint(245, 255), random.randint(245, 255))
        
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Metadata Container
    metadata = {
        "image_id": os.path.basename(filename),
        "drug_name": drug_data['name'],
        "diagnosis": drug_data['indi'],
        "is_trap": trap_mode,
        "is_visual_mismatch": False, 
        "is_consistent": True, 
        "has_leakage": False, 
        "masked_name": False,
        "quality_score": 1.0,
        "compliance_checked": True
    }

    # ==========================
    # BLOCK A: Header (Top Left)
    # ==========================
    off_x, off_y = get_jitter_offset(5)
    tx, ty = 50 + off_x, 30 + off_y
    draw.text((tx, ty), HOSPITAL_META["name"], fill="#003366", font=get_font(36, bold=True))
    draw.text((tx, ty+45), f"地址: {HOSPITAL_META['address']}", fill="black", font=get_font(18))
    draw.text((tx, ty+70), f"用藥諮詢: {HOSPITAL_META['phone']}", fill="red", font=get_font(24, bold=True))

    # ==========================
    # BLOCK B: Smart Tech (QR) (Top Right)
    # ==========================
    off_x, off_y = get_jitter_offset(5)
    
    # Anchor Point for QR Block
    qr_anchor_x = 650 + off_x
    qr_anchor_y = 25 + off_y
    
    qr_size = 110
    quiet_margin = 40 
    
    # Strict Forbidden Zone (White Patch)
    draw.rectangle([qr_anchor_x - quiet_margin, qr_anchor_y - quiet_margin, 
                    qr_anchor_x + qr_size + quiet_margin, qr_anchor_y + qr_size + quiet_margin + 60], 
                   fill="white", outline=None) 
    
    # Generate QR Data
    raw_name = "測試阿公"
    masked_name, is_masked = mask_privacy_info(raw_name, ratio=1.0) 
    metadata["masked_name"] = is_masked
    
    qr_data = {
        "nhi": drug_data['code'],
        "name": drug_data['name'], 
        "patient": masked_name, 
        "indi": drug_data['indi'],
    }
    
    qr_img = generate_real_qr_code(qr_data, size=qr_size)
    img.paste(qr_img, (qr_anchor_x, qr_anchor_y)) 
    
    # Labels attached to QR Block (Relative to Anchor)
    draw.text((qr_anchor_x, qr_anchor_y + 120), f"NHI: {drug_data['code']}", fill="black", font=get_font(16))
    draw.text((qr_anchor_x - 30, qr_anchor_y + 145), f"調劑: {HOSPITAL_META['pharmacist']}", fill="black", font=get_font(16))
    
    # Separator
    ly = 200 + random.randint(-5, 5); draw.line([(30, ly), (866, ly)], fill="#003366", width=4)
    
    # ==========================
    # BLOCK C: Patient Info
    # ==========================
    off_x, off_y = get_jitter_offset(5)
    px, py = 50 + off_x, 220 + off_y
    
    draw.text((px, py), f"姓名: {masked_name} (85歲)", fill="black", font=get_font(36, bold=True))
    
    today_str = datetime.date.today().strftime("%Y/%m/%d")
    draw.text((px+400, py+50), f"總量: 28 顆 (7天份)   日期: {today_str}", fill="black", font=get_font(24, bold=True))
    
    # Diagnosis
    diagnosis = drug_data['indi']
    if pair_type == "SOUND_ALIKE_CRITICAL" and trap_mode and "Lasix" in drug_data['name']:
         diagnosis = "胃潰瘍 (Gastric Ulcer)" 
         metadata["is_consistent"] = False
         
    metadata["diagnosis"] = diagnosis 
    draw.text((px, py+50), f"臨床診斷: {diagnosis}", fill="black", font=get_font(24))
    
    batch_y = py + 90
    draw.text((px, batch_y), f"許可證: {drug_data.get('license', 'N/A')}", fill="black", font=get_font(18))
    draw.text((px+400, batch_y), "批號: V26-HUMAN 效期: 2026/12", fill="#D32F2F", font=get_font(18, bold=True))

    # ==========================
    # BLOCK D: Drug Info (The Collision Zone)
    # ==========================
    off_x, off_y = get_jitter_offset(5)
    rx, ry = 40 + off_x, 360 + off_y
    
    # Background Box
    draw.rectangle([(rx, ry), (rx+816, ry+250)], fill="#F0F8FF") 
    
    text_x = rx + 20
    text_y = ry + 20
    
    # Dynamic Font Scaling for Name
    name_font, name_size = fit_text_to_width(draw, drug_data['name'], FONT_PATHS["Bold"], max_width=520, start_size=50) 
    
    draw.text((text_x, text_y), drug_data['name'], fill="black", font=name_font)
    draw.text((text_x, text_y + name_size + 15), drug_data['zh'], fill="#003366", font=get_font(40, bold=True))
    
    visual_desc = get_pill_description(drug_data)
    draw.text((text_x, text_y + name_size + 70), f"外觀: {visual_desc}", fill="black", font=get_font(22))
    
    # [V26] Dosage Instruction
    dosage = drug_data.get('dosage_instruction', '遵照醫囑服用')
    draw.text((text_x, text_y + name_size + 105), f"用法: {dosage}", fill="#006400", font=get_font(22, bold=True)) # DarkGreen
    
    # Pill Positioning logic
    pill_anchor_x = rx + 650 
    pill_anchor_y = ry + 50
    
    force_mismatch = (trap_mode and pair_type!="SOUND_ALIKE_CRITICAL")
    metadata["is_visual_mismatch"] = force_mismatch
    if force_mismatch: metadata["is_consistent"] = False
    
    draw_hyper_real_pill(draw, pill_anchor_x, pill_anchor_y, drug_data, force_mismatch=force_mismatch)
    
    # ==========================
    # BLOCK E: Usage Grid
    # ==========================
    off_x, off_y = get_jitter_offset(5)
    ux, uy = 50 + off_x, 620 + off_y
    
    draw.text((ux, uy), "服用方法 (Usage):", fill="black", font=get_font(28))
    
    x, y, w, h = ux, uy + 40, 800, 110
    draw.rectangle([x, y, x+w, y+h], outline="black", width=3)
    col_w = w // 4
    for i in range(1, 4): draw.line([x+i*col_w, y, x+i*col_w, y+h], fill="black", width=2)
    headers = ["早上", "中午", "晚上", "睡前"]
    
    is_before_meal = "飯前" in drug_data.get('warning', '')
    usage_code = drug_data.get('usage_code', 'BID')
    targets = [] 
    if "BID_MN" in usage_code: targets = [0, 1] # [V26 Fix] Morning + Noon
    elif "BID" in usage_code: targets = [0, 2]
    elif "TID" in usage_code: targets = [0, 1, 2]
    elif "QD" in usage_code: targets = [0]
    elif "HS" in usage_code: targets = [3]
    else: targets = [0, 2]
    
    for i in range(4):
        bx = x + i*col_w; cx = bx + col_w//2; cy = y + h//2
        draw.text((bx+10, y+5), headers[i], fill="black", font=get_font(24))
        icon_y = cy - 20
        if i == 0: draw_sun_rising(draw, cx, icon_y, 30) 
        elif i == 1: draw_sun_full(draw, cx, icon_y, 30)
        elif i == 2: draw_moon_evening(draw, cx, icon_y, 30)
        elif i == 3: draw_moon_sleeping(draw, cx, icon_y, 30)
        
        if i in targets and i != 3: 
            bowl_y = icon_y + 35
            if is_before_meal: draw_bowl_empty(draw, cx+25, bowl_y, 25)
            else: draw_bowl_full(draw, cx+25, bowl_y, 25)
            
        if i in targets:
            draw.ellipse([cx-25, cy+20, cx+25, cy+70], outline="red", width=4)
            draw.text((cx-10, cy+25), "1", fill="red", font=get_font(36, bold=True))
    
    # --- F. Warning ---
    off_x, off_y = get_jitter_offset(5)
    
    # [V24 Fix] Move Warning Box DOWN to 795
    wx, wy = 40 + off_x, 795 + off_y 
    
    draw.rectangle([(wx, wy), (wx+816, wy+75)], outline="#D32F2F", width=4)
    draw.text((wx+10, wy+10), "⚠️ 警語:", fill="#D32F2F", font=get_font(20, bold=True))
    
    # Clean Data
    raw_warning = drug_data['warning']
    clean_warning = raw_warning.replace("警語:", "").replace("警語：", "").strip()
    
    if "\n" in clean_warning:
        parts = clean_warning.split("\n")
        zh_text = parts[0].strip()
        en_text = parts[1].strip() if len(parts) > 1 else ""
    else:
        zh_text = clean_warning
        en_text = ""
        
    draw.text((wx+95, wy+10), zh_text, fill="black", font=get_font(20))
    if en_text:
        draw.text((wx+10, wy+40), en_text, fill="black", font=get_font(16))
    
    # Footer
    fx, fy = 40 + off_x, 880 + off_y
    quality_score = metadata["quality_score"]
    status = "TRAP" if trap_mode else "SAFE"
    draw.text((fx, fy), f"V26_HUMAN | Q={quality_score:.1f} | {status} | {filename[-8:]}", fill="black", font=get_font(12))

    # --- G. Post-Processing ---
    img = add_creases(img, intensity=0.5)
    img = apply_optical_stress(img, severity=1)
    img = apply_plastic_glare(img, intensity=0.6) 
    img = draw_watermark(img)
    
    img.save(filename)
    
    # Save Sidecar
    json_path = filename.replace(".jpg", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        
    print(f"✅ Generated V26: {filename}")

if __name__ == "__main__":
    download_fonts() 
    print("🚀 MedGemma V16 Expansion: Full Drug Library Started...")
    
    # ===== PHASE 1: Load Complete Drug Library =====
    print("\n📚 Loading Complete Drug Database from medgemma_data.py...")
    try:
        from medgemma_data import get_renderable_data
        LASA_PAIRS = get_renderable_data()
        total_drugs = sum(len(v) for v in LASA_PAIRS.values())
        print(f"✅ Loaded {total_drugs} drugs")
        for cat, drugs in LASA_PAIRS.items():
            print(f"   - {cat}: {len(drugs)} drugs")
    except ImportError:
        print("⚠️ medgemma_data.py not found, using hardcoded LASA_PAIRS")
        total_drugs = sum(len(v) for v in LASA_PAIRS.values())
    
    # ===== PHASE 2: Generate Variants =====
    import random
    import json
    from datetime import datetime
    
    random.seed(42)
    VARIANTS_PER_DRUG = 30
    TRAP_PROBABILITY = 0.3
    
    PATIENT_AGES = list(range(65, 95))
    PATIENT_NAMES = ["陳金龍", "林美玉", "張志明", "李建國", "王秀英", "黃明德", "劉淑芬", "吳文雄"]
    
    print(f"\n🏭 Generating {total_drugs} drugs × {VARIANTS_PER_DRUG} variants = {total_drugs * VARIANTS_PER_DRUG} samples\n")
    
    count = 0
    generated_files = []
    
    for cat, drugs in LASA_PAIRS.items():
        for drug_idx, d in enumerate(drugs):
            drug_shortname = d['name'].split()[0]
            print(f"🔄 [{cat}] {drug_shortname}...", end="")
            
            for variant_idx in range(VARIANTS_PER_DRUG):
                is_trap = random.random() < TRAP_PROBABILITY
                filename = f"{OUTPUT_DIR}/{cat}_{drug_shortname}_V{variant_idx:03d}.jpg"
                
                patient_age = random.choice(PATIENT_AGES)
                patient_name = random.choice(PATIENT_NAMES)
                
                # Select random parameters
                bg_color = random.choice(["white", "cream", "light_gray", "warm_white"])
                blur_level = random.choice([0.0, 0.5, 1.0]) # 0=Clear, 1=Blurry
                glare_intensity = random.choice([0.3, 0.6, 0.9])
                
                generate_v26_human_bag(
                    filename, cat, d, is_trap,
                    bg_color=bg_color,
                    blur_level=blur_level,
                    glare_intensity=glare_intensity
                )
                
                generated_files.append({
                    "filename": os.path.basename(filename),
                    "category": cat,
                    "drug_data": d,
                    "is_trap": is_trap,
                    "patient_age": patient_age,
                    "patient_name": patient_name,
                    "visual_params": {
                        "bg": bg_color,
                        "blur": blur_level,
                        "glare": glare_intensity
                    }
                })
                count += 1
            
            print(f" ✓ {VARIANTS_PER_DRUG} variants")
    
    print(f"\n🎉 Generated {count} samples in {OUTPUT_DIR}")
    
    # ===== PHASE 3: Generate Training JSON =====
    print("\n📦 Generating Training Dataset JSON...")
    
    dataset = []
    for idx, item in enumerate(generated_files):
        drug_name = item['drug_data']['name']
        is_trap = item['is_trap']
        patient_age = item['patient_age']
        dose_match = drug_name.split()[1] if len(drug_name.split()) > 1 else "N/A"
        
        if is_trap:
            if item['category'] == "SOUND_ALIKE_CRITICAL":
                status = "PHARMACIST_REVIEW_REQUIRED"
                reasoning = f"Step 1: Patient age {patient_age}. Drug {drug_name}. Step 2: SOUND-ALIKE category. Step 3: Verify with pharmacist. Ref: ISMP."
            elif patient_age >= 85:
                status = "ATTENTION_NEEDED"
                reasoning = f"Step 1: Elderly {patient_age}. Drug {drug_name}. Step 2: Beers 2023 dose reduction for >80. Step 3: Consult physician."
            else:
                status = "WARNING"
                reasoning = f"Step 1: Drug {drug_name}. Step 2: Visual mismatch. Step 3: Verify appearance."
        else:
            status = "WITHIN_STANDARD"
            reasoning = f"Step 1: Age {patient_age}. Drug {drug_name}. Step 2: Appropriate. Step 3: No critical issues."
        
        gpt_response = json.dumps({
            "extracted_data": {
                "patient": {"name": item['patient_name'], "age": patient_age},
                "drug": {"name": drug_name, "name_zh": item['drug_data']['zh'], "dose": dose_match},
                "usage": item['drug_data'].get('dosage_instruction', '遵照醫囑')
            },
            "safety_analysis": {"status": status, "reasoning": reasoning}
        }, ensure_ascii=False)
        
        dataset.append({
            "id": f"V16_{idx:05d}",
            "image": item['filename'],
            "difficulty": "hard" if is_trap else "easy",
            "risk_status": status,
            "conversations": [
                {"from": "human", "value": "You are 'SilverGuard CDS', a Clinical Decision Support System. Analyze this prescription image and output JSON with 'extracted_data' and 'safety_analysis'.\n<image>"},
                {"from": "gpt", "value": gpt_response}
            ]
        })
    
    random.shuffle(dataset)
    split_idx = int(len(dataset) * 0.9)
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]
    
    with open(f"{OUTPUT_DIR}/dataset_v16_train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(f"{OUTPUT_DIR}/dataset_v16_test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    with open(f"{OUTPUT_DIR}/dataset_v16_full.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Train: {len(train_data)} | Test: {len(test_data)} | Total: {len(dataset)}")
    print(f"\n" + "="*80)
    print(f"🎯 V16 Expansion Complete: {len(dataset)} samples ready!")
    print(f"="*80)
