"""
================================================================================
🏥 MedGemma Impact Challenge - Hardcore Training Pipeline (V17)
================================================================================
Author: Wang Yuan-dao (SilverGuard Project)
Goal:   Instruction Fine-Tuning (SFT) of MedGemma 1.5-4b-it
Method: 4-bit QLoRA + PEFT
Data:   V17 Hyper-Realistic Synthetic Dataset (30% Threat Injection)
================================================================================
"""

import os
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
import json
import random

# ==============================================================================
# 1. Configuration & Hyperparameters
# ==============================================================================
# Model ID
BASE_MODEL = "google/medgemma-1.5-4b-it" 
NEW_MODEL = "medgemma-1.5-silverguard-adapter"

# QLoRA Config
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

# LoRA Config
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training Arguments
output_dir = "./results"
num_train_epochs = 3 
fp16 = True
bf16 = False
per_device_train_batch_size = 2 # T4 fits 2-4
gradient_accumulation_steps = 4
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 50
logging_steps = 10

# SFT Parameters
max_seq_length = 512 # Drug bags have dense text
packing = False
device_map = {"": 0}

# ==============================================================================
# 2. Data Loading (V17 Dataset)
# ==============================================================================
print("📂 Loading Hyper-Realistic Dataset (V17)...")
dataset_path = os.getenv("MEDGEMMA_V17_DIR", "./assets/lasa_dataset_v17_compliance")
train_file = os.path.join(dataset_path, "dataset_v17_train.json")

# Fallback for Kaggle
candidates = [
    train_file,
    "/kaggle/working/assets/lasa_dataset_v17_compliance/dataset_v17_train.json",
    "./dataset_v17_train.json"
]

found_file = None
for c in candidates:
    if os.path.exists(c):
        found_file = c
        break

if not found_file:
    print(f"❌ Training Data Not Found! Checked: {candidates}")
    print("⚠️ Generate data first using 'generate_v17_fusion.py'")
    exit(1)

print(f"✅ Found Data at: {found_file}")

with open(found_file, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

print(f"✅ Loaded {len(raw_data)} samples. Inspecting first sample...")
# print(json.dumps(raw_data[0], indent=2, ensure_ascii=False))

# Transform to HuggingFace Dataset
# MedGemma expects Chat Format: <start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n...
def format_instruction(sample):
    # Construct System Prompt (Implicit in MedGemma or explicit)
    # MedGemma uses specific chat template
    
    ocr_text = sample.get('ocr_text', '')
    profile = sample.get('patient_profile', {})
    ground_truth = sample.get('ground_truth', {})

    instruction = f"""Analyze this drug bag image text for patient safety.
Extracted Text:
{ocr_text}

Patient Context:
{json.dumps(profile, ensure_ascii=False)}

Task:
1. Identify Drug Name, Dosage, Usage.
2. Check for Age/Gender contraindications.
3. Output JSON with safety status (PASS/WARNING/HIGH_RISK).
"""
    output = json.dumps(ground_truth, indent=2, ensure_ascii=False)
    
    # Gemma Format
    text = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{output}<end_of_turn>"
    return {"text": text}

# Convert list of dicts to HF Dataset
try:
    hf_dataset = Dataset.from_list(raw_data)
    hf_dataset = hf_dataset.map(format_instruction)
    print(f"✅ Dataset Formatted. Sample:\n{hf_dataset[0]['text'][:200]}...")
except Exception as e:
    print(f"⚠️ Dataset Conversion Failed: {e}")
    exit(1)

# ==============================================================================
# 3. Model Loading (4-bit Quantization)
# ==============================================================================
print("🤖 Loading MedGemma 1.5 (4-bit)...")
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

try:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
except Exception as e:
    print(f"❌ Model Load Failed: {e}")
    # Likely need HF Token login
    print("👉 Ensure you have logged in via `huggingface-cli login` or set HUGGINGFACE_TOKEN env var.")
    exit(1)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

#Config LoRA
print("🔧 Configuring LoRA Adapters...")
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=target_modules
)

# ==============================================================================
# 5. Training (SFT)
# ==============================================================================
print("🚀 Starting Training (SFT)...")
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="none" # Disable wandb for Kaggle cleanliness unless specifically needed
)

trainer = SFTTrainer(
    model=model,
    train_dataset=hf_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

trainer.train()
print("✅ Training Complete!")

# ==============================================================================
# 6. Save Model
# ==============================================================================
trainer.model.save_pretrained(NEW_MODEL)
print(f"💾 Adapter saved to {NEW_MODEL}")
