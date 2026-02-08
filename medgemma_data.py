"""
MedGemma Shared Drug Database (Source of Truth)
Extracted from: AI_Pharmacist_Guardian_V5.py
Purpose: Sync data between Training (V5), Generation (V16), and Stress Test.
"""

# [V8.8 Audit Fix] Global Safety Thresholds
# [Demo Recording] Blur Threshold Configuration
# Production: 100.0 (Conservative for Patient Safety)
# Demo Recording: 25.0 (Prevents false rejection from camera shake/phone photos)
BLUR_THRESHOLD = 25.0  # ⚠️ Set to 25.0 for smooth demo recording
# Note: Camera shake or phone photography typically scores 40-80
# A threshold of 100.0 would reject most demo recordings

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
        "warning": "胃潰瘍患者慎用。若有黑便請立即停藥就醫",
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
        "warning": "手術前5-7天需停藥。勿與其他抗凝血藥併用",
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
DRUG_ALIASES = {
    # Diabetes
    "glucophage": "metformin", "glucophage xr": "metformin", "fortamet": "metformin", "glumetza": "metformin",
    "amaryl": "glimepiride", "januvia": "sitagliptin", "daonil": "glibenclamide", "diamicron": "gliclazide",
    # Hypertension
    "norvasc": "amlodipine", "concor": "bisoprolol", "diovan": "valsartan", "dilatrend": "carvedilol", "lasix": "furosemide",
    # Sedative
    "stilnox": "zolpidem", "imovane": "zopiclone", "hydralazine": "hydralazine", "hydroxyzine": "hydroxyzine",
    # Cardiac
    "asa": "aspirin", "plavix": "clopidogrel", "aspirin": "aspirin", "bokey": "aspirin",
    # Analgesic
    "panadol": "acetaminophen", "acetaminophen": "acetaminophen",
    # Anticoagulant
    "coumadin": "warfarin", "warfarin": "warfarin", "xarelto": "rivaroxaban",
    # Lipid
    "lipitor": "atorvastatin", "crestor": "rosuvastatin",
}

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
            if "黃" in app: color = "yellow"
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
        # Changed DOKTER to "DOKTER / APOTEKER" for broader context
        "HIGH_RISK": "BAHAYA! JANGAN MINUM OBAT INI. HUBUNGI DOKTER ATAU APOTEKER SEKARANG.",
        "WARNING": "PERHATIAN. PERIKSA KEMBALI DOSISNYA.", # Check the dose again
        "SAFE": "OBAT INI AMAN. MINUM SESUAI RESEP."
    },
    "VIETNAMESE": {
        # Kept original (Perfect)
        "HIGH_RISK": "NGUY HIỂM! KHÔNG ĐƯỢC UỐNG THUỐC NÀY. GỌI BÁC SĨ NGAY.",
        "WARNING": "CHÚ Ý. KIỂM TRA LẠI LIỀU LƯỢNG VỚI BÁC SĨ.",
        "SAFE": "THUỐC NÀY AN TOÀN. UỐNG THEO TOA."
    },
    "TAIWANESE": {
        # Fixed "不通" -> "毋通" (Standard Hokkien)
        "HIGH_RISK": "危險！這藥毋通食，趕緊打電話問醫生。",
        "WARNING": "注意！這藥可能有問題，先問過醫生或是藥師。",
        "SAFE": "這藥沒問題，照醫生交代去食。"
    }
}
