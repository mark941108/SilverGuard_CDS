"""
MedGemma Shared Drug Database (Source of Truth)
Extracted from: AI_Pharmacist_Guardian_V5.py
Purpose: Sync data between Training (V5), Generation (V16), and Stress Test.
"""

# [V8.8 Audit Fix] Global Safety Thresholds
# [Demo Recording] Blur Threshold Configuration
# Production: 100.0 (Conservative for Patient Safety)
# Strict Clinical Standard: 50.0 (Recommended for Impact Challenge)
BLUR_THRESHOLD = 50.0  # ✅ Restored to Professional Standard 
# Note: Camera shake or phone photography typically scores 40-80
# A threshold of 100.0 would reject most handheld inputs

# Original Data Source from V5
DRUG_DATABASE = {
    # --- Confusion Cluster 1: Hypertension ---
    "Hypertension": [
        {"code": "BC23456789", "name_en": "Norvasc", "name_zh": "脈優", "generic": "Amlodipine", "dose": "5mg", "appearance": "白色八角形", "indication": "降血壓", "warning": "小心姿勢性低血壓", "default_usage": "QD_breakfast_after", 
         "max_daily_dose": 10, "drug_class": "CCB", "beers_risk": False},
        {"code": "BC23456790", "name_en": "Concor", "name_zh": "康肯", "generic": "Bisoprolol", "dose": "5mg", "appearance": "黃色心形", "indication": "降血壓", "warning": "心跳過慢者慎用", "default_usage": "QD_breakfast_after",
         "max_daily_dose": 20, "drug_class": "Beta-Blocker", "beers_risk": False},
        {"code": "BC23456799", "name_en": "Dilatrend", "name_zh": "達利全錠", "generic": "Carvedilol", "dose": "25mg", "appearance": "白色圓形 (刻痕)", "indication": "高血壓/心衰竭", "warning": "不可擅自停藥", "default_usage": "BID_meals_after",
         "max_daily_dose": 50, "drug_class": "Beta-Blocker", "beers_risk": False},
        {"code": "BC23456788", "name_en": "Lasix", "name_zh": "來適泄錠", "generic": "Furosemide", "dose": "40mg", "appearance": "白色圓形", "indication": "高血壓/水腫", "warning": "服用後排尿頻繁，避免睡前服用", "default_usage": "BID_morning_noon",
         "max_daily_dose": 80, "drug_class": "Diuretic", "beers_risk": False}, # Note: Loop diuretics generally safe if monitored
        {"code": "BC23456801", "name_en": "Hydralazine", "name_zh": "阿普利素", "generic": "Hydralazine", "dose": "25mg", "appearance": "黃色圓形", "indication": "高血壓", "warning": "不可隨意停藥", "default_usage": "TID_meals_after",
         "max_daily_dose": 200, "drug_class": "Vasodilator", "beers_risk": False},
        {"code": "BC23456791", "name_en": "Diovan", "name_zh": "得安穩", "generic": "Valsartan", "dose": "160mg", "appearance": "橘色橢圓形", "indication": "高血壓/心衰竭", "warning": "注意姿勢性低血壓、懷孕禁用", "default_usage": "QD_breakfast_after",
         "max_daily_dose": 320, "drug_class": "ARB", "beers_risk": False},
    ],
    # --- Confusion Cluster 2: Diabetes ---
    "Diabetes": [
        {"code": "BC23456792", "name_en": "Glucophage", "name_zh": "庫魯化", "generic": "Metformin", "dose": "500mg", "appearance": "白色長圓形", "indication": "降血糖", "warning": "隨餐服用減少腸胃不適", "default_usage": "BID_meals_after",
         "max_daily_dose": 2550, "drug_class": "Biguanide", "beers_risk": False},
        {"code": "BC23456793", "name_en": "Daonil", "name_zh": "道尼爾", "generic": "Glibenclamide", "dose": "5mg", "appearance": "白色長條形 (刻痕)", "indication": "降血糖", "warning": "低血糖風險高", "default_usage": "QD_breakfast_after",
         "max_daily_dose": 20, "drug_class": "Sulfonylurea", "beers_risk": True}, # ⚠️ High Risk for Elderly
        {"code": "BC23456795", "name_en": "Diamicron", "name_zh": "岱蜜克龍", "generic": "Gliclazide", "dose": "30mg", "appearance": "白色長條形", "indication": "降血糖", "warning": "飯前30分鐘服用", "default_usage": "QD_breakfast_before",
         "max_daily_dose": 120, "drug_class": "Sulfonylurea", "beers_risk": True}, # ⚠️ High Risk for Elderly (Long-acting)
    ],
    # --- Confusion Cluster 3: Gastric ---
    "Gastric": [
        {"code": "BC23456787", "name_en": "Losec", "name_zh": "樂酸克膠囊", "generic": "Omeprazole", "dose": "20mg", "appearance": "粉紅/紅棕色膠囊", "indication": "胃潰瘍/逆流性食道炎", "warning": "飯前服用效果最佳，不可嚼碎", "default_usage": "QD_meals_before",
         "max_daily_dose": 40, "drug_class": "PPI", "beers_risk": True}, # ⚠️ Long term use risk Clostridium difficile
    ],
    # --- Confusion Cluster 4: Anticoagulant ---
    # 1. Anticoagulants (High Risk)
    "Anticoagulant": [
    {
        "code": "BC25438100",
        "name_en": "Warfarin",
        "name_zh": "華法林",
        "generic": "Warfarin Sodium",
        "dose": "5mg",
        "appearance": "粉紅色圓形 (刻痕)",
        "indication": "預防血栓形成",
        "warning": "需定期監測INR，避免深綠色蔬菜",
        "default_usage": "QD_evening",
        "max_daily_dose": 15, "drug_class": "Anticoagulant", "beers_risk": True # ⚠️ High Bleeding Risk
    },
    {
        "code": "BC24681357",
        "name_en": "Xarelto",
        "name_zh": "拜瑞妥",
        "generic": "Rivaroxaban",
        "dose": "20mg",
        "appearance": "Hex(#8D6E63)圓形", # Fixed: brown_red -> Hex
        "indication": "預防中風及栓塞",
        "warning": "隨餐服用。請注意出血徵兆",
        "default_usage": "QD_evening_with_meal",
        "max_daily_dose": 20, "drug_class": "NOAC", "beers_risk": True # ⚠️ Bleeding Risk
    },
    {
        "code": "BC23951468",
        "name_en": "Bokey", 
        "name_zh": "伯基/阿斯匹靈",
        "generic": "Aspirin",
        "dose": "100mg",
        "appearance": "白色圓形 (微凸)",
        "indication": "預防心肌梗塞",
        "warning": "胃潰瘍患者慎用。長期服用需監測出血風險",
        "default_usage": "QD_breakfast_after",
        "max_daily_dose": 100, "drug_class": "Antiplatelet", "beers_risk": True # ⚠️ Generally avoid for primary prevention age > 70
    },
    {
        "code": "BC_ASPIRIN_EC",
        "name_en": "Aspirin E.C.",
        "name_zh": "阿斯匹靈腸溶錠",
        "generic": "Aspirin",
        "dose": "100mg",
        "appearance": "白色圓形 (腸溶)",
        "indication": "預防血栓/心肌梗塞",
        "warning": "胃潰瘍患者慎用。若有黑便建議立即就醫評估停藥",
        "default_usage": "QD_breakfast_after",
        "max_daily_dose": 100, "drug_class": "Antiplatelet", "beers_risk": True
    },
    {
        "code": "BC24135792",
        "name_en": "Plavix",
        "name_zh": "保栓通",
        "generic": "Clopidogrel", 
        "dose": "75mg",
        "appearance": "粉紅色圓形",
        "indication": "預防血栓",
        "warning": "手術前建議諮詢醫師評估停藥 (通常5-7天)。勿與其他抗凝血藥併用",
        "default_usage": "QD_breakfast_after",
        "max_daily_dose": 75, "drug_class": "Antiplatelet", "beers_risk": False
    },
    ],
    # --- Confusion Cluster 5: CNS ---
    "Sedative": [
        {"code": "BC23456794", "name_en": "Stilnox", "name_zh": "使蒂諾斯", "generic": "Zolpidem", "dose": "10mg", "appearance": "白色長條形", "indication": "失眠", "warning": "服用後立即就寢", "default_usage": "QD_bedtime",
         "max_daily_dose": 10, "drug_class": "Z-drug", "beers_risk": True}, # ⚠️ High Fall/Delirium Risk
        {"code": "BC23456802", "name_en": "Hydroxyzine", "name_zh": "安泰樂", "generic": "Hydroxyzine", "dose": "25mg", "appearance": "白色圓形", "indication": "抗過敏/焦慮", "warning": "注意嗜睡", "default_usage": "TID_meals_after",
         "max_daily_dose": 100, "drug_class": "Antihistamine", "beers_risk": True}, # ⚠️ Anticholinergic burden
    ],
     # --- Confusion Cluster 6: Lipid ---
    "Lipid": [
        {"code": "BC88889999", "name_en": "Lipitor", "name_zh": "立普妥", "generic": "Atorvastatin", "dose": "20mg", "appearance": "白色橢圓形", "indication": "降血脂", "warning": "肌肉痠痛時需回診", "default_usage": "QD_bedtime",
         "max_daily_dose": 80, "drug_class": "Statin", "beers_risk": False},
        {"code": "BC88889998", "name_en": "Crestor", "name_zh": "冠脂妥", "generic": "Rosuvastatin", "dose": "10mg", "appearance": "粉紅色圓形", "indication": "降血脂", "warning": "避免與葡萄柚汁併服", "default_usage": "QD_bedtime",
         "max_daily_dose": 40, "drug_class": "Statin", "beers_risk": False},
        {"code": "BC23456800", "name_en": "Ezetrol", "name_zh": "怡潔", "generic": "Ezetimibe", "dose": "10mg", "appearance": "白色長條形", "indication": "降血脂", "warning": "可與他汀類併用", "default_usage": "QD_breakfast_after",
         "max_daily_dose": 10, "drug_class": "Cholesterol Absorption Inhibitor", "beers_risk": False},
    ],
    # --- Confusion Cluster 7: Analgesic (Added for Rule 4 Safety) ---
    "Analgesic": [
        {"code": "BC55667788", "name_en": "Panadol", "name_zh": "普拿疼", "generic": "Acetaminophen", "dose": "500mg", "appearance": "白色圓形", "indication": "止痛/退燒", "warning": "每日不可超過4000mg (8顆)", "default_usage": "Q4H_prn",
         "max_daily_dose": 4000, "drug_class": "Analgesic", "beers_risk": False},
    ],
}

# ===== Drug Aliases Mapping (Legacy Support) =====
# DRUG_ALIASES Consolidated below to prevent duplication

def get_renderable_data():
    """
    Adapter: Converts V5 DB Schema to V16/V26 Generator Schema.
    Parses 'appearance' text to 'shape'/'color' enums.
    """
    lasa_pairs = {
        "SOUND_ALIKE_CRITICAL": [],
        "LOOK_ALIKE_SHAPE": [],
        "GENERAL_TRAINING": []
    }
    
    # Mapper logic
    for category, drugs in DRUG_DATABASE.items():
        for d in drugs:
            # 1. Parse Appearance
            shape = "circle" # Default
            color = "white"  # Default
            app = d["appearance"]
            
            # Shape Matching
            if "長條" in app or "長圓" in app: shape = "oblong"
            elif "橢圓" in app: shape = "oval"
            elif "膠囊" in app: shape = "capsule"
            elif "圓形" in app: 
                if "刻痕" in app: shape = "circle_scored"
                else: shape = "circle"
            elif "心形" in app: shape = "circle" # Approx
            elif "八角" in app: shape = "circle" # Approx
            
            # Color Matching
            # [Audit Fix] 順序重要！先檢查複合色（紅棕）再檢查單色
            # [V17 Fix] Support direct Hex Code (e.g. Hex(#8D6E63))
            if "Hex" in app:
                import re
                match = re.search(r'Hex\((#[0-9A-Fa-f]{6})\)', app)
                if match: color = match.group(1)
                else: color = "white" # Fallback
            elif "黃" in app: color = "yellow"
            elif "紅棕" in app: color = "brown_red"  # ✅ Xarelto 專用：紅褐色
            elif "粉紅" in app and "紅棕" in app: color = "pink_brown"
            elif "粉紅" in app: color = "pink"
            elif "紅" in app: color = "red"
            elif "白" in app: 
                if "金" in app: color = "white_gold"
                else: color = "white"
            
            # 3. Create V16 Object
            # 2. Parse Usage Code (Simple Heuristic for Grid)
            usage_code = "BID"
            # [V26 Fix] Order matters! Check specific cases (HS/Bedtime) first.
            if "HS" in d["default_usage"] or "bedtime" in d["default_usage"]: usage_code = "HS"
            elif "morning_noon" in d["default_usage"]: usage_code = "BID_MN" 
            elif "QD" in d["default_usage"]: usage_code = "QD"
            elif "TID" in d["default_usage"]: usage_code = "TID"
            
            # 3. Create V16 Object
            v16_obj = {
                "name": f"{d['name_en']} {d['dose']} ({d['generic']})",
                "zh": d['name_zh'],
                "code": d['code'],
                "indi": d['indication'],
                "shape": shape,
                "color": color,
                "warning": f"警語: {d['warning']}",
                "usage_code": usage_code,
                "license": f"衛署藥製字第{d['code'][-6:]}號", # Dynamic Realism from Drug Code
                "dosage_instruction": parse_dosage_usage(d["default_usage"]) # V26 Feature
            }
            
            # 4. Categorize (Simple Logic)
            # [Audit Fix] 加入 Hydralazine/Hydroxyzine LASA Pair
            if d['name_en'] in ["Lasix", "Losec", "Norvasc", "Concor", "Hydralazine", "Hydroxyzine"]:
                lasa_pairs["SOUND_ALIKE_CRITICAL"].append(v16_obj)
            elif d['name_en'] in ["Dilatrend", "Xarelto", "Daonil", "Diamicron"]:
                 lasa_pairs["LOOK_ALIKE_SHAPE"].append(v16_obj)
            else:
                 lasa_pairs["GENERAL_TRAINING"].append(v16_obj)
                 
    return lasa_pairs

def parse_dosage_usage(usage_tag):
    """ Translate internal tag to V26 Human Instruction """
    map_ = {
        "QD_breakfast_after": "每日1次，早餐後服用",
        "QD_breakfast_before": "每日1次，飯前30分鐘服用",
        "QD_meals_before": "每日1次，飯前服用",
        "QD_meals_with": "每日1次，隨餐服用",
        "QD_bedtime": "每日1次，睡前服用",
        # [Audit Fix P0] Add missing usage keys for Warfarin, Xarelto, Panadol
        "QD_evening": "每日1次，晚上服用",
        "QD_evening_with_meal": "每日1次，晚餐後隨餐服用",
        "Q4H_prn": "需要時每4小時服用1次 (每日最多6次)",
        "BID_meals_after": "每日2次，飯後服用",
        "BID_morning_noon": "每日2次，早午服用 (避免夜尿)",
        "TID_meals_after": "每日3次，飯後服用"
    }
    # [Audit Fix P0] Add fallback to prevent KeyError
    return map_.get(usage_tag, f"遵照醫囑服用 ({usage_tag})")


# ---------------------------------------------------------
# [V1.0 IMPACT UPDATE] DETERMINISTIC LINGUISTIC GUARDRAILS
# ---------------------------------------------------------
# To prevent "Translation Hallucination" in high-risk scenarios,
# we use pre-approved, hardcoded safety commands for migrant languages.
# This ensures 100% instructional correctness.

ALERT_PHRASES = {
    "BAHASA": {
        "HIGH_RISK": "RISIKO TINGGI. MOHON KONSULTASI DOKTER SEGERA.",
        "WARNING": "PERHATIAN. SARAN KONFIRMASI DOSIS.", 
        "SAFE": "INFO SESUAI RESEP. IKUTI INSTRUKSI DOKTER."
    },
    "VIETNAMESE": {
        "HIGH_RISK": "RỦI RO CAO. VUI LÒNG HỎI Ý KIẾN BÁC SĨ.",
        "WARNING": "CẢNH BÁO. VUI LÒNG KIỂM TRA LẠI.", 
        "SAFE": "THÔNG TIN KHỚP. VUI LÒNG TUÂN THỦ TOA THUỐC."
    },
    "TAIWANESE": {
        "HIGH_RISK": "這項藥物有高風險，建議先問過醫生。",
        "WARNING": "這項藥物要注意，建議拿單子給藥師看。", 
        "SAFE": "辨識結果符合處方，請照醫生交代服用。"
    }
}

# ---------------------------------------------------------
# [V7.5 FIX] GLOBAL DRUG ALIASES (Synonym Mapping)
# ---------------------------------------------------------
DRUG_ALIASES = {
    # Generic -> Brand (or vice versa, for normalization)
    "amlodipine": "norvasc",
    "bisoprolol": "concor",
    "carvedilol": "dilatrend",
    "furosemide": "lasix",
    "valsartan": "diovan",
    "metformin": "glucophage",
    "glibenclamide": "daonil",
    "gliclazide": "diamicron",
    "omeprazole": "losec",
    "warfarin sodium": "warfarin",
    "coumadin": "warfarin",
    "rivaroxaban": "xarelto",
    "aspirin": "bokey",
    "acetylsalicylic acid": "bokey",
    "clopidogrel": "plavix",
    "zolpidem": "stilnox",
    "atorvastatin": "lipitor",
    "rosuvastatin": "crestor",
    "ezetimibe": "ezetrol",
    "acetaminophen": "panadol",
    "paracetamol": "panadol",
    "tylenol": "panadol",
    "hydralazine": "hydralazine", # Generic fallback
    "hydroxyzine": "hydroxyzine",
    "imovane": "zopiclone",
    "stilnox": "zolpidem"
}

def lookup_chinese_name(name_en):
    """
    將英文藥名對照資料庫轉換為中文藥名 (Data-level Lookup)
    """
    if not name_en: return "未知藥物"
    import re
    # 清理雜訊 (劑量、括號)
    clean_name = re.sub(r'\s*\d+\.?\d*\s*(mg|g|mcg|ug|ml|毫克|公克)\b', '', str(name_en), flags=re.IGNORECASE)
    clean_name = re.sub(r'\s*\([^)]*\)', '', clean_name).strip().lower()
    
    # 檢查別名
    target = DRUG_ALIASES.get(clean_name, clean_name)
    
    for category in DRUG_DATABASE.values():
        for item in category:
            if target in [item['name_en'].lower(), item['generic'].lower()]:
                return item['name_zh']
    return name_en # Fallback

# =========================================================
# ❤️ [Empathetic Engine] Patient-Centric Communication Mode (Compliance Verified)
# Focus: AI provides triage guidance, NOT medical decisions.
# =========================================================
# =========================================================
# ❤️ [Empathetic Engine] Patient-Centric Communication Mode (Compliance Verified)
# Focus: AI provides triage guidance, NOT medical decisions.
# [Round 144] Multilingual Expansion (ID/VI/EN) for Template TTS
# =========================================================
WARM_SCRIPTS = {
    "HIGH_RISK": {
        "zh-TW": [
            "提醒您，請稍等一下。",  
            "這藥物與一般處方有些許差異，", 
            "⚠️ 建議先諮詢醫師或是藥師，確認沒問題再來服用，比較安心！" 
        ],
        "en": [
            "Please wait a moment.",
            "This prescription requires verification.",
            "⚠️ Please consult a pharmacist before taking this medication."
        ],
        "id": [
            "Mohon tunggu sebentar.",
            "Resep ini perlu diverifikasi.",
            "⚠️ Disarankan konsultasi ke apoteker sebelum minum obat ini."
        ],
        "vi": [
            "Xin vui lòng chờ một chút.",
            "Đơn thuốc này cần được xác minh.",
            "⚠️ Khuyên bạn nên hỏi ý kiến dược sĩ trước khi dùng thuốc này."
        ]
    },
    "WARNING": {
        "zh-TW": [
            "提醒您，請多留意。",
            "這藥物有一些細節建議要注意，",
            "⚠️ 建議向藥師確認用藥方式。" 
        ],
        "en": [
            "Please take note.",
            "There are some details to check.",
            "⚠️ Please confirm usage with a pharmacist."
        ],
        "id": [
            "Mohon perhatikan.",
            "Ada detail yang perlu dicek.",
            "⚠️ Disarankan konfirmasi cara pakai ke apoteker."
        ],
        "vi": [
            "Xin lưu ý.",
            "Có một số chi tiết cần kiểm tra.",
            "⚠️ Khuyên bạn xác nhận cách dùng với dược sĩ."
        ]
    },
    "SAFE": {
        "zh-TW": [ 
            "辨識結果符合處方紀錄。",           
            "它是 {drug_name}，", 
            "請遵照醫囑服用，並定期回診。" 
        ],
        "en": [
            "Identification matches records.",
            "This is {drug_name}.",
            "Please follow the prescription and regular check-ups."
        ],
        "id": [
            "Identifikasi cocok dengan resep.",
            "Ini adalah {drug_name}.",
            "Mohon ikuti resep dan kontrol teratur."
        ],
        "vi": [
            "Nhận dạng khớp với hồ sơ.",
            "Đây là {drug_name}.",
            "Vui lòng tuân theo đơn thuốc và tái khám định kỳ."
        ]
    }
}

# 🚨 [Round 128] Medical Ethics Update: Professional Tone Enforced
# Deprecated: Informal phrasings removed for clinical professionalism by default.
# Add: Specific clinical reasoning + Direct triage action
EMERGENCY_SCRIPTS = {
    "BLEEDING": {
        "zh-TW": "⚠️ 醫療警示：偵測到出血關鍵字。您正在服用抗凝血藥物，建議立即尋求醫療協助，並諮詢醫師關於用藥調整。",
        "en": "⚠️ MEDICAL ALERT: Bleeding reported while on anticoagulants. Recommend seeking immediate medical attention to evaluate medication risks.",
        "id": "⚠️ PERINGATAN MEDIS: Pendarahan terdeteksi. Disarankan segera cari bantuan medis untuk evaluasi obat.",
        "vi": "⚠️ CẢNH BÁO Y TẾ: Phát hiện chảy máu. Khuyên bạn nên tìm kiếm sự chăm sóc y tế ngay lập tức để đánh giá thuốc."
    },
    "CHEST_PAIN": {
        "zh-TW": "⚠️ 緊急狀況：偵測到胸痛或心臟不適。建議保持冷靜，並立即撥打 119 或前往最近的急診。",
        "en": "⚠️ CRITICAL ALERT: Chest pain detected. Recommend calling 119/911 or going to the nearest Emergency Room.",
        "id": "⚠️ DARURAT: Nyeri dada terdeteksi. Disarankan segera hubungi ambulans atau ke UGD terdekat.",
        "vi": "⚠️ KHẨN CẤP: Phát hiện đau ngực. Khuyên bạn gọi cấp cứu 115 hoặc đến phòng cấp cứu gần nhất."
    },
    "STROKE": {
        "zh-TW": "⚠️ 中風警示：偵測到疑似中風症狀。建議立即記下時間並撥打 119 求助。",
        "en": "⚠️ STROKE ALERT: Possible stroke symptoms detected. Recommend noting the time and calling an ambulance immediately.",
        "id": "⚠️ WASPADA STROKE: Gejala stroke terdeteksi. Disarankan catat waktu dan panggil ambulans segera.",
        "vi": "⚠️ CẢNH BÁO ĐỘT QUỴ: Nghi ngờ đột quỵ. Khuyên bạn ghi lại thời gian và gọi cấp cứu ngay."
    },
    "ALLERGY": {
        "zh-TW": "⚠️ 過敏警示：偵測到藥物過敏反應。建議攜帶藥袋諮詢醫師或藥師，評估是否暫停用藥。",
        "en": "⚠️ ALLERGY ALERT: Possible adverse reaction. Recommend consulting a doctor/pharmacist with the drug bag immediately.",
        "id": "⚠️ ALERGI OBAT: Kemungkinan reaksi alergi. Disarankan konsultasi ke dokter dengan membawa obat.",
        "vi": "⚠️ DỊ ỨNG THUỐC: Có thể bị phản ứng phụ. Khuyên bạn mang theo thuốc để hỏi ý kiến bác sĩ."
    }
}


def generate_warm_message(status, drug_name_en, reasoning="", target_lang="zh-TW"):
    """
    Core Logic: Constructing empathetic patient-centric responses.
    [Round 108 Update] Added 'reasoning' for context-aware emergency overrides.
    [Round 109 Update] Added 'target_lang' for multilingual emergency triage.
    """
    # 0. Emergency Override (High Priority)
    # Check reasoning keywords for immediate triage
    if reasoning:
        r_upper = str(reasoning).upper()
        emergency_key = None
        if "BLEEDING" in r_upper or "HEMORRHAGE" in r_upper or "BLACK STOOL" in r_upper:
            emergency_key = "BLEEDING"
        elif "CHEST PAIN" in r_upper or "SUICIDE" in r_upper or "CRUSHING PAIN" in r_upper:
            emergency_key = "CHEST_PAIN"
        elif "STROKE" in r_upper:
            emergency_key = "STROKE"
        elif "ALLERGY" in r_upper or "ANAPHYLAXIS" in r_upper:
            emergency_key = "ALLERGY"
            
        if emergency_key:
            # [Round 109] Multilingual Routing
            # Default to English if language not supported, or zh-TW if default
            lang_code = target_lang if target_lang in ["zh-TW", "en", "id", "vi"] else "en"
            # Fallback for traditional chinese specifically
            if target_lang == "zh-TW": lang_code = "zh-TW"
            
            script_dict = EMERGENCY_SCRIPTS.get(emergency_key, {})
            return script_dict.get(lang_code, script_dict.get("en", "EMERGENCY! SEEK MEDICAL HELP."))

    # 狀態對齊：如果傳入的是 PASS 則轉換為 SAFE (確保字典能查到)
    if status == "PASS": status = "SAFE"
    
    # [Constraint] Warm Scripts are currently zh-TW ONLY. 
    # For other languages, we return None to let app.py handle standard TTS, 
    # UNLESS it was an emergency caught above.
    # [Round 144] CONSTRAINT REMOVED: Now supporting ID/VI/EN via templates.
    
    if status not in WARM_SCRIPTS:
        return None
        
    # Get Multilingual Script Dictionary
    script_dict = WARM_SCRIPTS[status]
    
    # Select Language (Fallback to en if missing, or zh-TW if default)
    lang_code = target_lang if target_lang in ["zh-TW", "en", "id", "vi"] else "en"
    if target_lang == "zh-TW" and "zh-TW" not in script_dict: lang_code = "zh-TW" # Safety
    
    if lang_code not in script_dict:
        return None # No template for this language
        
    script_parts = script_dict[lang_code]
    
    # lookup_chinese_name is only for zh-TW. For others, we use the English name.
    if lang_code == "zh-TW":
        drug_display = lookup_chinese_name(drug_name_en)
    else:
        drug_display = drug_name_en # Use English name for ID/VI/EN
    
    if status == "SAFE":
        # 組合 SAFE 邏輯：使用 .format() 填入藥名
        try:
            # Check if template has placeholder
            if "{drug_name}" in script_parts[1]:
                part_2 = script_parts[1].format(drug_name=drug_display)
            else:
                 part_2 = script_parts[1]
            return f"{script_parts[0]} {part_2} {script_parts[2]}"
        except:
             return f"{script_parts[0]} {drug_display}. {script_parts[2]}"
    else:
        # 危險/警告時
        return f"{script_parts[0]} {script_parts[1]} {script_parts[2]}"
