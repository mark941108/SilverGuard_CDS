"""
MedGemma Shared Drug Database (Source of Truth)
Extracted from: AI_Pharmacist_Guardian_V5.py
Purpose: Sync data between Training (V5), Generation (V16), and Stress Test.
"""

# Original Data Source from V5
DRUG_DATABASE = {
    # --- Confusion Cluster 1: Hypertension ---
    "Hypertension": [
        {"code": "BC23456789", "name_en": "Norvasc", "name_zh": "脈優", "generic": "Amlodipine", "dose": "5mg", "appearance": "白色八角形", "indication": "降血壓", "warning": "小心姿勢性低血壓", "default_usage": "QD_breakfast_after"},
        {"code": "BC23456790", "name_en": "Concor", "name_zh": "康肯", "generic": "Bisoprolol", "dose": "5mg", "appearance": "黃色心形", "indication": "降血壓", "warning": "心跳過慢者慎用", "default_usage": "QD_breakfast_after"},
        {"code": "BC23456799", "name_en": "Dilatrend", "name_zh": "達利全錠", "generic": "Carvedilol", "dose": "25mg", "appearance": "白色圓形 (刻痕)", "indication": "高血壓/心衰竭", "warning": "不可擅自停藥", "default_usage": "BID_meals_after"},
        {"code": "BC23456788", "name_en": "Lasix", "name_zh": "來適泄錠", "generic": "Furosemide", "dose": "40mg", "appearance": "白色圓形", "indication": "高血壓/水腫", "warning": "服用後排尿頻繁，避免睡前服用", "default_usage": "BID_morning_noon"},
    ],
    # --- Confusion Cluster 2: Diabetes ---
    "Diabetes": [
        {"code": "BC23456792", "name_en": "Glucophage", "name_zh": "庫魯化", "generic": "Metformin", "dose": "500mg", "appearance": "白色長圓形", "indication": "降血糖", "warning": "隨餐服用減少腸胃不適", "default_usage": "BID_meals_after"},
        {"code": "BC23456793", "name_en": "Daonil", "name_zh": "道尼爾", "generic": "Glibenclamide", "dose": "5mg", "appearance": "白色長條形 (刻痕)", "indication": "降血糖", "warning": "低血糖風險高", "default_usage": "QD_breakfast_after"},
        {"code": "BC23456795", "name_en": "Diamicron", "name_zh": "岱蜜克龍", "generic": "Gliclazide", "dose": "30mg", "appearance": "白色長條形", "indication": "降血糖", "warning": "飯前30分鐘服用", "default_usage": "QD_breakfast_before"},
    ],
    # --- Confusion Cluster 3: Gastric ---
    "Gastric": [
        {"code": "BC23456787", "name_en": "Losec", "name_zh": "樂酸克膠囊", "generic": "Omeprazole", "dose": "20mg", "appearance": "粉紅/紅棕色膠囊", "indication": "胃潰瘍/逆流性食道炎", "warning": "飯前服用效果最佳，不可嚼碎", "default_usage": "QD_meals_before"},
    ],
    # --- Confusion Cluster 4: Anticoagulant ---
    "Anticoagulant": [
         {"code": "BC23456786", "name_en": "Xarelto", "name_zh": "拜瑞妥膜衣錠", "generic": "Rivaroxaban", "dose": "15mg", "appearance": "紅色圓形", "indication": "預防中風/血栓", "warning": "隨餐服用。請注意出血徵兆", "default_usage": "QD_meals_with"},
         {"code": "BC77778888", "name_en": "Warfarin", "name_zh": "可化凝", "generic": "Warfarin", "dose": "5mg", "appearance": "粉紅色圓形", "indication": "抗凝血", "warning": "需定期監測INR，避免深綠色蔬菜", "default_usage": "QD_bedtime"},
         {"code": "BC55556666", "name_en": "Aspirin", "name_zh": "阿斯匹靈", "generic": "ASA", "dose": "100mg", "appearance": "白色圓形", "indication": "預防血栓", "warning": "胃潰瘍患者慎用", "default_usage": "QD_breakfast_after"},
         {"code": "BC55556667", "name_en": "Plavix", "name_zh": "保栓通", "generic": "Clopidogrel", "dose": "75mg", "appearance": "粉紅色圓形", "indication": "預防血栓", "warning": "手術前需停藥", "default_usage": "QD_breakfast_after"},
    ],
    # --- Confusion Cluster 5: CNS ---
    "Sedative": [
        {"code": "BC23456794", "name_en": "Stilnox", "name_zh": "使蒂諾斯", "generic": "Zolpidem", "dose": "10mg", "appearance": "白色長條形", "indication": "失眠", "warning": "服用後立即就寢", "default_usage": "QD_bedtime"},
        {"code": "BC23456801", "name_en": "Hydralazine", "name_zh": "阿普利素", "generic": "Hydralazine", "dose": "25mg", "appearance": "黃色圓形", "indication": "高血壓", "warning": "不可隨意停藥", "default_usage": "TID_meals_after"},
        {"code": "BC23456802", "name_en": "Hydroxyzine", "name_zh": "安泰樂", "generic": "Hydroxyzine", "dose": "25mg", "appearance": "白色圓形", "indication": "抗過敏/焦慮", "warning": "注意嗜睡", "default_usage": "TID_meals_after"},
    ],
     # --- Confusion Cluster 6: Lipid ---
    "Lipid": [
        {"code": "BC88889999", "name_en": "Lipitor", "name_zh": "立普妥", "generic": "Atorvastatin", "dose": "20mg", "appearance": "白色橢圓形", "indication": "降血脂", "warning": "肌肉痠痛時需回診", "default_usage": "QD_bedtime"},
        {"code": "BC88889998", "name_en": "Crestor", "name_zh": "冠脂妥", "generic": "Rosuvastatin", "dose": "10mg", "appearance": "粉紅色圓形", "indication": "降血脂", "warning": "避免與葡萄柚汁併服", "default_usage": "QD_bedtime"},
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
    "asa": "aspirin", "plavix": "clopidogrel", "aspirin": "aspirin",
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
            if "黃" in app: color = "yellow"
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
                "license": "衛署藥製字第000000號", # Placeholder if missing
                "dosage_instruction": parse_dosage_usage(d["default_usage"]) # V26 Feature
            }
            
            # 4. Categorize (Simple Logic)
            if d['name_en'] in ["Lasix", "Losec", "Norvasc", "Concor"]:
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
        "BID_meals_after": "每日2次，飯後服用",
        "BID_morning_noon": "每日2次，早午服用 (避免夜尿)",
        "TID_meals_after": "每日3次，飯後服用"
    }
    return map_.get(usage_tag, "遵照醫囑服用")
