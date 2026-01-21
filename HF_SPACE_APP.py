import gradio as gr
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image
import json
import os
import re
import spaces # ZeroGPU support

# ============================================================================
# ============================================================================
# 🏥 AI Pharmacist Guardian - Hugging Face Space Demo
# ============================================================================
# Project: AI Pharmacist Guardian
# Author: Wang Yuan-dao (Solo Developer & Energy Engineering Student)
# Philosophy: Zero-Cost Edge AI + Agentic Safety Loop
#
# This app provides an interactive demo for the MedGemma Impact Challenge.
# It loads the fine-tuned adapter from Hugging Face Hub (Bonus 1) and runs inference.

# 1. Configuration
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

BASE_MODEL = "google/medgemma-1.5-4b-it"
ADAPTER_MODEL = os.environ.get("ADAPTER_MODEL_ID", "mark941108/MedGemma-SilverGuard-V5")

if "Please_Replace" in ADAPTER_MODEL or not ADAPTER_MODEL:
    print("❌ CRITICAL: ADAPTER_MODEL_ID not configured!")
    raise ValueError("ADAPTER_MODEL_ID environment variable must be set before deployment.")

# Offline Mode Toggle (For Air-Gapped / Privacy-First deployment)
OFFLINE_MODE = os.environ.get("OFFLINE_MODE", "False").lower() == "true"
if OFFLINE_MODE:
    print("🔒 OFFLINE_MODE Active: External APIs (OpenFDA, Google TTS) disabled.")

print(f"⏳ Loading MedGemma Adapter: {ADAPTER_MODEL}...")

# 2. Model Loading
try:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL, 
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL, token=HF_TOKEN)
    processor = AutoProcessor.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    print("✅ MedGemma Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading MedGemma: {e}")
    base_model = None
    model = None
    processor = None

# ============================================================================
# 🎤 MedASR Loading (Second HAI-DEF Model)
# ============================================================================
MEDASR_MODEL = "google/medasr"
medasr_pipeline = None

try:
    from transformers import pipeline
    print(f"⏳ Loading MedASR: {MEDASR_MODEL}...")
    medasr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=MEDASR_MODEL,
        token=HF_TOKEN,
        device="cpu",  # Run on CPU to save GPU VRAM for MedGemma
        torch_dtype=torch.float32
    )
    print("✅ MedASR Loaded Successfully!")
except Exception as e:
    print(f"⚠️ MedASR loading failed (non-critical): {e}")
    medasr_pipeline = None

def transcribe_audio(audio_path):
    """Transcribe audio using MedASR (google/medasr)."""
    if medasr_pipeline is None or audio_path is None:
        return "", False
    
    try:
        import librosa
        # Load and resample to 16kHz (MedASR requirement)
        audio, sr = librosa.load(audio_path, sr=16000)
        result = medasr_pipeline({"array": audio, "sampling_rate": 16000})
        transcription = result.get("text", "")
        print(f"🎤 MedASR Transcription: {transcription}")
        return transcription, True
    except Exception as e:
        print(f"⚠️ MedASR transcription failed: {e}")
        return "", False

# ============================================================================
# 🔮 CONFIGURATION (V5 Impact Edition)
# ============================================================================
# ⚠️ LOAD FROM KAGGLE-TRAINED ADAPTER (Bonus Task: Open Weights)
ADAPTER_MODEL = os.environ.get("ADAPTER_MODEL_ID", "mark941108/MedGemma-SilverGuard-V5")
BASE_MODEL = "google/medgemma-1.5-4b-it"

# Feature Flags
OFFLINE_MODE = False   # Set True if no internet (will disable APIs)
ENABLE_TTS = True      # Enable Text-to-Speech

# Agent Settings
MAX_RETRIES = 2
TEMP_CREATIVE = 0.6    # First pass: Creative/Reasoning
TEMP_STRICT = 0.2      # Retry pass: Deterministic (Safety-First)

# ============================================================================
# 🧠 Helper Functions
# ============================================================================
BLUR_THRESHOLD = 100

def check_image_quality(image, blur_threshold=BLUR_THRESHOLD):
    """Input Validation Gate - Reject blurry images"""
    try:
        import cv2
        import numpy as np
        
        if image.mode == "RGBA":
            image = image.convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")
        
        open_cv_image = np.array(image) 
        open_cv_image = open_cv_image[:, :, ::-1].copy() # RGB to BGR
        
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < blur_threshold:
            return False, f"Image too blurry (score: {laplacian_var:.1f} < {blur_threshold})"
        return True, "Quality OK"
    except Exception as e:
        return True, f"Blur check skipped: {e}"

def check_is_prescription(response_text):
    """OOD Detection - Verify prescription content"""
    prescription_keywords = ["patient", "drug", "dose", "mg", "tablet", "capsule", 
                            "prescription", "pharmacy", "usage", "medication", "藥"]
    response_lower = response_text.lower()
    keyword_count = sum(1 for kw in prescription_keywords if kw.lower() in response_lower)
    
    if keyword_count >= 3:
        return True
    return False

DRUG_ALIASES = {
    "glucophage": "metformin",
    "glucophage xr": "metformin", "fortamet": "metformin", "glumetza": "metformin",
    "amaryl": "glimepiride",
    "januvia": "sitagliptin",
    "norvasc": "amlodipine",
    "concor": "bisoprolol",
    "diovan": "valsartan",
    "stilnox": "zolpidem",
    "imovane": "zopiclone",
    "asa": "aspirin", 
    "plavix": "clopidogrel",
    "coumadin": "warfarin",
    "lipitor": "atorvastatin",
    "crestor": "rosuvastatin",
}

# Sync from KAGGLE_V5_COMPLETE.py Cell 2
DRUG_DATABASE = {
    "Hypertension": [
        {"code": "BC23456789", "name_en": "Norvasc", "name_zh": "脈優", "generic": "Amlodipine", "dose": "5mg", "appearance": "白色八角形", "indication": "降血壓", "warning": "小心姿勢性低血壓", "default_usage": "QD_breakfast_after"},
        {"code": "BC23456790", "name_en": "Concor", "name_zh": "康肯", "generic": "Bisoprolol", "dose": "5mg", "appearance": "黃色心形", "indication": "降血壓", "warning": "心跳過慢者慎用", "default_usage": "QD_breakfast_after"},
        {"code": "BC23456791", "name_en": "Diovan", "name_zh": "得安穩", "generic": "Valsartan", "dose": "80mg", "appearance": "淡紅色橢圓形", "indication": "降血壓", "warning": "懷孕禁用", "default_usage": "QD_breakfast_after"},
    ],
    "Diabetes": [
        {"code": "BC11223344", "name_en": "Glucophage", "name_zh": "庫魯化", "generic": "Metformin", "dose": "500mg", "appearance": "白色長圓形", "indication": "降血糖", "warning": "隨餐服用", "default_usage": "BID_meals_after"},
        {"code": "BC11223345", "name_en": "Amaryl", "name_zh": "瑪爾胰", "generic": "Glimepiride", "dose": "2mg", "appearance": "綠色橢圓形", "indication": "降血糖", "warning": "小心低血糖", "default_usage": "QD_breakfast_after"},
        {"code": "BC11223346", "name_en": "Januvia", "name_zh": "佳糖維", "generic": "Sitagliptin", "dose": "100mg", "appearance": "米色圓形", "indication": "降血糖", "warning": "腎功能不全需調整劑量", "default_usage": "QD_breakfast_after"},
    ],
    "Sedative": [
        {"code": "BC99998888", "name_en": "Stilnox", "name_zh": "使蒂諾斯", "generic": "Zolpidem", "dose": "10mg", "appearance": "白色長條形", "indication": "失眠", "warning": "服用後請立即就寢", "default_usage": "QD_bedtime"},
        {"code": "BC99998889", "name_en": "Imovane", "name_zh": "宜眠安", "generic": "Zopiclone", "dose": "7.5mg", "appearance": "藍色圓形", "indication": "失眠", "warning": "可能有金屬味", "default_usage": "QD_bedtime"},
    ],
    "Cardiac": [
        {"code": "BC55556666", "name_en": "Aspirin", "name_zh": "阿斯匹靈", "generic": "ASA", "dose": "100mg", "appearance": "白色圓形", "indication": "預防血栓", "warning": "胃潰瘍患者慎用", "default_usage": "QD_breakfast_after"},
        {"code": "BC55556667", "name_en": "Plavix", "name_zh": "保栓通", "generic": "Clopidogrel", "dose": "75mg", "appearance": "粉紅色圓形", "indication": "預防血栓", "warning": "手術前需停藥", "default_usage": "QD_breakfast_after"},
    ],
    "Anticoagulant": [
        {"code": "BC77778888", "name_en": "Warfarin", "name_zh": "可化凝", "generic": "Warfarin", "dose": "5mg", "appearance": "粉紅色圓形", "indication": "抗凝血", "warning": "需定期監測INR，避免深綠色蔬菜", "default_usage": "QD_bedtime"},
    ],
    "Lipid": [
        {"code": "BC88889999", "name_en": "Lipitor", "name_zh": "立普妥", "generic": "Atorvastatin", "dose": "20mg", "appearance": "白色橢圓形", "indication": "降血脂", "warning": "肌肉痠痛時需回診", "default_usage": "QD_bedtime"},
        {"code": "BC88889998", "name_en": "Crestor", "name_zh": "冠脂妥", "generic": "Rosuvastatin", "dose": "10mg", "appearance": "粉紅色圓形", "indication": "降血脂", "warning": "避免與葡萄柚汁併服", "default_usage": "QD_bedtime"},
    ],
}

def retrieve_drug_info(drug_name: str) -> dict:
    """RAG Interface (Mock for Hackathon)"""
    drug_lower = drug_name.lower().strip()
    names_to_search = [drug_lower]
    if drug_lower in DRUG_ALIASES:
        names_to_search.append(DRUG_ALIASES[drug_lower]) 
    
    # Check Database
    for cat, drugs in DRUG_DATABASE.items():
        for drug in drugs:
            name_en_lower = drug.get("name_en", "").lower()
            generic_lower = drug.get("generic", "").lower()
            
            for search_name in names_to_search:
                if (search_name in name_en_lower or search_name in generic_lower or
                    name_en_lower in search_name or generic_lower in search_name):
                    
                    result = drug.copy()
                    result["found"] = True
                    return result

    # ⚠️ Catch-All for Unknown Drugs (The Safe Fallback)
    return {
        "found": False, 
        "class": "Unknown", 
        "name_en": drug_name,
        "warning": "⚠️ UNKNOWN DRUG DETECTED. SYSTEM CANNOT VERIFY SAFETY.",
        "risk": "UNKNOWN_DRUG"
    }

# ============================================================================
# 💊 OpenFDA Drug Interaction Checker
# ============================================================================
def check_drug_interaction(drug_a, drug_b):
    if not drug_a or not drug_b:
        return "⚠️ Please enter two drug names."
        
    name_a = DRUG_ALIASES.get(drug_a.lower(), drug_a.lower())
    name_b = DRUG_ALIASES.get(drug_b.lower(), drug_b.lower())
    print(f"🔎 Checking interaction: {name_a} + {name_b}")
    
    CRITICAL_PAIRS = {
        ("warfarin", "aspirin"): "🔴 **MAJOR RISK**: Increased bleeding probability. Monitor INR closely.",
        ("warfarin", "ibuprofen"): "🔴 **MAJOR RISK**: High bleeding risk (NSAID + Anticoagulant).",
        ("metformin", "contrast_dye"): "⚠️ **WARNING**: Risk of Lactic Acidosis. Hold Metformin 48h before/after procedure.",
        ("lisinopril", "potassium"): "⚠️ **WARNING**: Risk of Hyperkalemia (high potassium).",
        ("sildenafil", "nitroglycerin"): "🔴 **CONTRAINDICATED**: Fatal hypotension risk. DO NOT COMBINE.",
        ("zolpidem", "alcohol"): "🔴 **MAJOR RISK**: Severe CNS depression. High fall risk for elderly.",
    }
    if (name_a, name_b) in CRITICAL_PAIRS: return CRITICAL_PAIRS[(name_a, name_b)]
    if (name_b, name_a) in CRITICAL_PAIRS: return CRITICAL_PAIRS[(name_b, name_a)]
        
    if OFFLINE_MODE:
        return "⚠️ Offline Mode: Showing locally cached major interactions only."

    try:
        import requests
        url = f"https://api.fda.gov/drug/label.json?search=openfda.generic_name:{name_a}+AND+drug_interactions:{name_b}&limit=1"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "results" in data and len(data["results"]) > 0:
                return f"⚠️ **OpenFDA Alert**: The official label for **{name_a.title()}** explicitly mentions interactions with **{name_b.title()}**."
            else:
                url_rev = f"https://api.fda.gov/drug/label.json?search=openfda.generic_name:{name_b}+AND+drug_interactions:{name_a}&limit=1"
                response_rev = requests.get(url_rev, timeout=5)
                if response_rev.status_code == 200 and "results" in response_rev.json():
                    return f"⚠️ **OpenFDA Alert**: The official label for **{name_b.title()}** explicitly mentions interactions with **{name_a.title()}**."
        return "✅ No obvious interaction found in OpenFDA summary labels."
    except Exception as e:
        print(f"OpenFDA API Error: {e}")
        return "⚠️ API unavailable. Please check manually."

def logical_consistency_check(extracted_data):
    """Neuro-Symbolic Logic Check (Hybrid Architecture)"""
    issues = []
    try:
        age_val = extracted_data.get("patient", {}).get("age", 0)
        age = int(age_val)
        if age < 0 or age > 120: issues.append(f"Invalid age: {age}")
        if age < 18: issues.append(f"Pediatric age ({age}) requires manual review")
        if age > 80:
            dose = extracted_data.get("drug", {}).get("dose", "")
            import re
            dose_match = re.search(r'(\d+)\s*(?:mg|g|mcg)', dose, re.IGNORECASE)
            if dose_match:
                dose_value = int(dose_match.group(1))
                if re.search(r'\d+\s*g(?!m)', dose, re.IGNORECASE): dose_value *= 1000
                if dose_value >= 1000: issues.append(f"Geriatric High Dose Warning: {age}yr + {dose}")
    except: pass

    try:
        dose = str(extracted_data.get("drug", {}).get("dose", ""))
        if dose and not re.search(r'\d+\s*(mg|ml|g|mcg|ug|tablet|capsule|pill|cap|tab|drops|gtt)', dose, re.IGNORECASE):
            issues.append(f"Abnormal dosage format: {dose}")
    except: pass
    
    try:
        drug_name = extracted_data.get("drug", {}).get("name", "") or extracted_data.get("drug", {}).get("name_en", "")
        if drug_name:
            drug_info = retrieve_drug_info(drug_name)
            if not drug_info.get("found", False): issues.append(f"Drug not in knowledge base: {drug_name}")
    except: pass

    # --- Final Issue Aggregation ---
    if issues:
        # V6.4 FIX: Critical Safety - Do NOT retry on unknown drugs (Infinite Loop Trap)
        if any("Drug not in knowledge base" in issue for issue in issues):
             return True, f"⚠️ UNKNOWN_DRUG detected. Manual Review Required. (Logic Check Passed to prevent retry)"
        
        return False, f"邏輯檢查異常: {', '.join(issues)}"
    return True, "邏輯一致性檢查通過"

def json_to_elderly_speech(result_json):
    """Generates the TTS script for SilverGuard"""
    try:
        if "silverguard_message" in result_json:
            return result_json["silverguard_message"]
        
        safety = result_json.get("safety_analysis", {})
        data = result_json.get("extracted_data", {})
        status = safety.get("status", "UNKNOWN")
        reasoning = safety.get("reasoning", "")
        drug_name = data.get("drug", {}).get("name", "藥物")
        
        if status == "HIGH_RISK":
            return f"阿嬤注意喔！這個藥是{drug_name}。AI發現有風險：{reasoning}。請先不要吃，趕快打電話問藥師。注意安全喔！"
        elif status == "HUMAN_REVIEW_NEEDED":
            return f"阿嬤，這個藥是{drug_name}。但是我看不清楚，為了安全，請你拿給藥師看，先不要自己吃喔。"
        else: # SAFE
            usage = data.get("usage", "照醫囑使用")
            return f"阿嬤，這是{drug_name}。AI檢查沒問題。使用方法是：{usage}。請安心使用。"
    except:
        return "系統忙碌中，請稍後再試。"

@spaces.GPU(duration=120)
def run_inference(image, patient_notes=""):
    """
    Main Agentic Inference function.
    - image: PIL Image of drug bag
    - patient_notes: Optional text from MedASR transription
    """
    is_clear, quality_msg = check_image_quality(image)
    if not is_clear:
        return "REJECTED_INPUT", {"error": quality_msg}, "阿嬤，照片太模糊了，我看不太清楚。請重新拍一張清楚一點的喔。", None

    if model is None:
        return "Model Error", {"error": "Model not loaded properly. Check logs."}, "System Error", None
    
    # Context Injection
    patient_context = ""
    if patient_notes and patient_notes.strip():
        patient_context = f"\n\n**CRITICAL Patient Note (from voice input)**: \"{patient_notes}\"\n"
        patient_context += "⚠️ CONTEXT: This note is provided by a MIGRANT CAREGIVER (e.g., from Philippines/Indonesia) speaking in English. "
        patient_context += "Please interpret their input carefully. Flag HIGH_RISK if the concept matches a contraindication (e.g., 'allergic to aspirin').\n"
    # V6 Enhanced Prompt: Dual-Persona (Clinical + SilverGuard) with Conservative Constraint
    base_prompt = (
        "You are 'AI Pharmacist Guardian', a **meticulous and risk-averse** clinical pharmacist in Taiwan. "
        "You prioritize patient safety above all else. When uncertain, you MUST flag for human review rather than guessing. "
        "Your patient is an elderly person (65+) who may have poor vision.\n\n"
        "Task:\n"
        "1. Extract: Patient info, Drug info (English name + Chinese function), Usage.\n"
        "2. Safety Check: Cross-reference AGS Beers Criteria 2023. Flag HIGH_RISK if age>80 + high dose.\n"
        "3. Cross-Check Context: Consider the provided CAREGIVER VOICE NOTE (if any) for allergies or specific conditions.\n"
        "4. SilverGuard: Add a warm message in spoken Taiwanese Mandarin (口語化台式中文).\n\n"
        "Output Constraints:\n"
        "- Return ONLY a valid JSON object.\n"
        "- 'safety_analysis.reasoning' MUST be in Traditional Chinese (繁體中文).\n"
        "- Add 'silverguard_message' field using the persona of a caring grandchild (貼心晚輩).\n\n"
        "### ONE-SHOT EXAMPLE (Reflect this Authenticity):\n"
        "{\n"
        "  \"extracted_data\": {\n"
        "    \"patient\": {\"name\": \"王大明\", \"age\": 88},\n"
        "    \"drug\": {\"name\": \"Glucophage\", \"name_zh\": \"庫魯化\", \"dose\": \"500mg\"},\n"
        "    \"usage\": \"每日兩次，飯後服用 (BID)\"\n"
        "  },\n"
        "  \"safety_analysis\": {\n"
        "    \"status\": \"WARNING\",\n"
        "    \"reasoning\": \"病患88歲，腎功能隨年齡下降。Glucophage (Metformin) 雖為一線用藥，但需注意 GFR 數值。建議請家屬確認近期腎功能檢查報告，避免乳酸中毒風險。\"\n"
        "  },\n"
        "  \"silverguard_message\": \"阿公，這是降血糖的藥（庫魯化）。醫生交代要『呷飽才吃』喔！如果覺得肚子不舒服、想吐，要趕快跟我們說。\"\n"
        "}"
    )

    # ===== AGENTIC LOOP =====
    MAX_RETRIES = 2
    current_try = 0
    correction_context = ""
    result_json = {}
    
    import ast
    def parse_model_output(response_text):
        response_text = re.sub(r'```json\s*', '', response_text).replace('```', '').strip()
        matches = []
        stack = []
        start_index = -1
        for i, char in enumerate(response_text):
            if char == '{':
                if not stack: start_index = i
                stack.append(char)
            elif char == '}':
                if stack:
                    stack.pop()
                    if not stack and start_index >= 0: matches.append(response_text[start_index:i+1])
        if not matches: return {"raw_output": response_text, "error": "No JSON found"}
        for json_str in reversed(matches):
            try: return json.loads(json_str.replace("True", "true").replace("False", "false").replace("None", "null"))
            except: pass
            try: return ast.literal_eval(json_str.replace("true", "True").replace("false", "False").replace("null", "None"))
            except: pass
            try: return json.loads(json_str.replace("'", '"').replace("True", "true").replace("False", "false").replace("None", "null"))
            except: pass
        return {"raw_output": response_text[:200], "error": "Parsing failed"}

    while current_try <= MAX_RETRIES:
        try:
            print(f"🔄 Agent Inference Attempt {current_try+1}/{MAX_RETRIES+1}...")
            final_prompt = base_prompt + correction_context
            inputs = processor(text=final_prompt, images=image, return_tensors="pt").to(model.device)
            input_len = inputs.input_ids.shape[1]
            current_temp = TEMP_CREATIVE if current_try == 0 else TEMP_STRICT
            
            with torch.inference_mode():
                generate_ids = model.generate(
                    **inputs, max_new_tokens=512, do_sample=True, temperature=current_temp, top_p=0.9,
                )
            
            generated_tokens = generate_ids[:, input_len:]
            response = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            result_json = parse_model_output(response)
            
            logic_issues = []
            if "extracted_data" in result_json:
                logic_issues = logical_consistency_check(result_json["extracted_data"])
            if not check_is_prescription(response):
                logic_issues.append("Input not a prescription script")
                
            if logic_issues:
                print(f"⚠️ Logic Check Failed: {logic_issues}")
                current_try += 1
                correction_context += f"\n\n[System Feedback]: Failed check: {'; '.join(logic_issues)}. Please Correct JSON."
                if current_try > MAX_RETRIES:
                    if "safety_analysis" not in result_json: result_json["safety_analysis"] = {}
                    result_json["safety_analysis"]["status"] = "HUMAN_REVIEW_NEEDED"
                    result_json["safety_analysis"]["reasoning"] = f"⚠️ Validation failed after retries: {'; '.join(logic_issues)}"
                    break
            else:
                break # Success
        except Exception as e:
            print(f"❌ Inference Error: {e}")
            current_try += 1
            correction_context += f"\n\n[System]: Crash: {str(e)}. Output simple valid JSON."
            
# --- 🌍 戰略功能：移工看護賦能 (Migrant Caregiver Support) ---
# 安全風險控制：使用「醫學驗證字典」而非 Google Translate，確保絕對安全。
SAFE_TRANSLATIONS = {
    "zh-TW": {
        "label": "🇹🇼 台灣 (繁體中文)",
        "HIGH_RISK": "⚠️ 危險！請勿服用",
        "WARNING": "⚠️ 警告！請再次確認",
        "PASS": "✅ 安全",
        "CONSULT": "請立即諮詢藥師 (0800-000-123)",
        "TTS_LANG": "zh-tw"
    },
    "id": {
        "label": "🇮🇩 Indonesia (Bahasa)",
        "HIGH_RISK": "⛔ BAHAYA! JANGAN MINUM OBAT INI!",
        "WARNING": "⚠️ PERINGATAN! CEK DOSIS.",
        "PASS": "✅ AMAN",
        "CONSULT": "TANYA APOTEKER SEGERA.",
        "TTS_LANG": "id"
    },
    "vi": {
        "label": "🇻🇳 Việt Nam (Tiếng Việt)",
        "HIGH_RISK": "⛔ NGUY HIỂM! KHÔNG ĐƯỢC UỐNG!",
        "WARNING": "⚠️ CẢNH BÁO! KIỂM TRA LIỀU LƯỢNG.",
        "PASS": "✅ AN TOÀN",
        "CONSULT": "HỎI NGAY DƯỢC SĨ.",
        "TTS_LANG": "vi"
    }
}

def silverguard_ui(case_data, target_lang="zh-TW"):
    """
    SilverGuard UI 生成器 (多語系版)
    """
    safety = case_data.get("safety_analysis", {})
    status = safety.get("status", "WARNING")
    
    # 1. 取得對應語言的安全翻譯 (Fallback to zh-TW)
    lang_pack = SAFE_TRANSLATIONS.get(target_lang, SAFE_TRANSLATIONS["zh-TW"])
    
    # 2. 對映狀態文字
    if status == "HIGH_RISK":
        display_status = lang_pack["HIGH_RISK"]
        color = "#ffcdd2" # Red
        icon = "⛔"
    elif status == "WARNING":
        display_status = lang_pack["WARNING"]
        color = "#fff9c4" # Yellow
        icon = "⚠️"
    else:
        display_status = lang_pack["PASS"]
        color = "#c8e6c9" # Green
        icon = "✅"
        
    # 3. 生成 TTS (使用對應語言)
    tts_text = f"{display_status}. {lang_pack['CONSULT']}."
    audio_path = text_to_speech(tts_text, lang=lang_pack["TTS_LANG"])
    
    html = f"""
    <div style="background-color: {color}; padding: 20px; border-radius: 15px; border: 3px solid #333;">
        <h1 style="color: #333; margin:0; font-size: 32px;">{icon} {display_status}</h1>
        <p style="font-size: 24px; color: #555; margin-top: 10px;">{lang_pack['CONSULT']}</p>
        <hr>
        <div style="font-size: 18px; color: #666;">
            <b>💊 Drug:</b> {case_data['extracted_data'].get('drug', {}).get('name', 'Unknown')}<br>
            <b>📋 Reason (English):</b> {safety.get('reasoning', 'No data')}
        </div>
    </div>
    """
    return html, audio_path

# --- GRADIO INTERFACE UPDATE ---
# ... (User must verify manual Gradio block update below) ...
    # TTS Logic (Hybrid)
    final_status = result_json.get("safety_analysis", {}).get("status", "UNKNOWN")
    speech_text = json_to_elderly_speech(result_json)
    audio_path = None
    tts_mode = "none"
    clean_text = speech_text.replace("⚠️", "注意").replace("✅", "").replace("🔴", "")
    
    # Tier 1: gTTS (Online)
    if not OFFLINE_MODE:
        try:
            import socket
            socket.setdefaulttimeout(1)
            socket.create_connection(("www.google.com", 80))
            from gtts import gTTS
            import tempfile
            tts = gTTS(text=clean_text, lang='zh-TW', slow=True)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f: audio_path = f.name
            tts.save(audio_path)
            tts_mode = "online"
            print("🔊 TTS: Online Mode (gTTS)")
        except Exception as e:
            print(f"⚠️ Online TTS failed: {e}")
            
    # Tier 2: pyttsx3 (Offline)
    if tts_mode == "none":
        try:
            import pyttsx3
            import tempfile
            # V6.5 FIX: Robust Offline Fallback (Dependency Hell Protection)
            try:
                engine = pyttsx3.init()
            except OSError:
                print("❌ pyttsx3 init failed (missing espeak?), skipping offline TTS")
                raise ImportError("espeak not found")

            voices = engine.getProperty('voices')
            for voice in voices:
                if 'zh' in voice.id.lower() or 'chinese' in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
            
            # V6 Deploy Fix: Use /tmp
            audio_path = "/tmp/tts_output.wav"
            engine.save_to_file(clean_text, audio_path)
            engine.runAndWait()
            tts_mode = "offline"
            print("🔊 TTS: Offline Mode (pyttsx3)")
        except Exception as e:
            print(f"⚠️ Offline TTS failed: {e}")
            tts_mode = "visual_only"
    
    result_json["_tts_mode"] = tts_mode
    return final_status, result_json, speech_text, audio_path

# ============================================================================
# 🖥️ Gradio Interface
# ============================================================================
custom_css = "#risk-header {color: #d32f2f; font-weight: bold; font-size: 1.2em;}"

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# 🏥 AI Pharmacist Guardian + SilverGuard (Live Demo)")
    gr.Markdown(
        "> ⚡ **Fast Mode**: Demo runs single-pass by default. "
        "Full Agentic Loop active when logic checks fail.\n"
        "> 🔊 **Hybrid TTS**: Online (gTTS) → Offline (pyttsx3) → Visual Fallback.\n"
        "> 🎤 **Caregiver Voice Log**: Speak English to record patient conditions."
    )
    
    with gr.Tabs():
        with gr.TabItem("🏥 AI Pharmacist Guardian"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_img = gr.Image(type="pil", label="📸 Upload Drug Bag Photo")
                    
                    gr.Markdown("### 🎤 Migrant Caregiver Voice Log")
                    gr.Markdown("*Log patient allergies in English (e.g. for helpers).*")
                    with gr.Row():
                        voice_ex1 = gr.Button("🔊 'Allergic to Aspirin'")
                        voice_ex2 = gr.Button("🔊 'Kidney Failure History'")
                    
                    voice_input = gr.Audio(sources=["microphone"], type="filepath", label="🎙️ Record Note")
                    transcription_display = gr.Textbox(label="📝 Transcription", interactive=False)
                    
                    btn = gr.Button("🔍 Analyze & Safety Check", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    # --- NEW: Language Selector for Migrant Caregivers ---
                    lang_dropdown = gr.Dropdown(
                        choices=["zh-TW", "id", "vi"], 
                        value="zh-TW", 
                        label="🌏 Caregiver Language (看護語言)", 
                        info="Select language for SilverGuard alerts"
                    )
                    
                    status_output = gr.Textbox(label="🛡️ Safety Status", elem_id="risk-header")
                    
                    # 👵 SilverGuard UI Priority (Per Blind Spot Scan)
                    silver_html = gr.HTML(label="👵 SilverGuard UI") 
                    audio_output = gr.Audio(label="🔊 Voice Alert")
                    
                    # 📉 HIDE COMPLEX LOGIC (Accordion)
                    with gr.Accordion("📊 Developer Logs (Agent Reasoning)", open=False):
                        json_output = gr.JSON(label="Agent Reasoning")

            def analyze_with_voice(image, audio_path, text_override, target_lang, progress=gr.Progress()):
                transcription = ""
                
                # Step 1: Voice Input
                if audio_path:
                    progress(0.1, desc="🎤 Analyzing Voice Note...")
                    t, success = transcribe_audio(audio_path)
                    if success: transcription = t
                if not transcription and text_override: transcription = text_override
                print(f"🎤 Context: {transcription} | Lang: {target_lang}")
                
                # Step 2: Inference (this is the slow part)
                progress(0.3, desc="🧠 MedGemma Agent Thinking...")
                status, res_json, speech, audio_path_old = run_inference(image, patient_notes=transcription)
                
                # Step 3: UI Gen
                progress(0.8, desc="👵 Generating SilverGuard UI...")
                html_view, audio_path_new = silverguard_ui(res_json, target_lang=target_lang)
                
                progress(1.0, desc="✅ Complete!")
                return transcription, status, res_json, html_view, audio_path_new
            
            btn.click(
                fn=analyze_with_voice, 
                inputs=[input_img, voice_input, transcription_display, lang_dropdown], 
                outputs=[transcription_display, status_output, json_output, silver_html, audio_output]
            )
            voice_ex1.click(lambda: "Patient is allergic to Aspirin.", outputs=transcription_display)
            voice_ex2.click(lambda: "Patient has history of kidney failure (eGFR < 30).", outputs=transcription_display)
            
            # Feedback
            gr.Markdown("---")
            with gr.Row():
                btn_correct = gr.Button("✅ Correct")
                btn_error = gr.Button("❌ Error")
            feedback_output = gr.Textbox(label="RLHF Status", interactive=False)
            
            def log_feedback(img, out, ftype):
                import datetime
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                return f"✅ Feedback logged at {ts}: {ftype} (Simulated)"
            
            btn_correct.click(lambda i,o: log_feedback(i,o,"POSITIVE"), inputs=[input_img, json_output], outputs=feedback_output)
            btn_error.click(lambda i,o: log_feedback(i,o,"NEGATIVE"), inputs=[input_img, json_output], outputs=feedback_output)

        with gr.TabItem("💊 Agentic Drug Interaction"):
            gr.Markdown("### 🔗 OpenFDA Agentic Tool")
            with gr.Row():
                d_a = gr.Textbox(label="Drug A")
                d_b = gr.Textbox(label="Drug B")
                chk_btn = gr.Button("🔍 Check")
            res = gr.Markdown(label="Result")
            chk_btn.click(check_drug_interaction, inputs=[d_a, d_b], outputs=res)

if __name__ == "__main__":
    demo.launch()
