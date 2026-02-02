---
language:
- en
- zh
license: cc-by-4.0
library_name: peft
tags:
- medical
- vlm
- agentic
- safety
---

# Model Card for MedGemma Guardian (V1.0 Impact Edition)

## 📋 Model Details

MedGemma Guardian is a specialized **Agentic Vision-Language Model (VLM)** designed to act as a safety layer for medication verification. It is a fine-tuned version of Google's [MedGemma 1.5 4B](https://huggingface.co/google/medgemma-1.5-4b-it), optimized for edge deployment using QLoRA.

-   **Developed by:** Yuan-dao Wang (MedGemma Impact Challenge)
-   **Model Type:** Multi-modal Vision-Language Model (VLM) based on Gemma 3 architecture.
-   **Language(s):** English, Traditional Chinese (Taiwan Localization)
-   **License:** CC-BY 4.0
-   **Finetuned from model:** `google/medgemma-1.5-4b-it`
-   **Training Method:** QLoRA (4-bit quantization) with Rank 16, Alpha 32.

## 🎯 Intended Use

This model is designed to assist licensed pharmacists and caregivers in verifying prescription safety.

-   **Primary Use Case:** Extracting medication details (Drug Name, Dosage, Frequency) from drug bag images and validating them against patient age and clinical guidelines.
-   **Intended Users:** Pharmacists, Clinical Staff, and Caregivers (via SilverGuard interface).
-   **Out of Scope:** This model is **NOT** a diagnostic tool. It should not be used to prescribe medication or replace professional medical judgment.

## ⚠️ Limitations & Known Issues

This is a **research prototype** and has several important limitations:

1.  **Training Data (Sim2Real Gap):**
    The model was trained primarily on **synthetic drug bag images** generated to mimic Taiwanese hospital standards. While we applied adversarial augmentation (Sim2Real techniques), performance on extremely noisy real-world images may vary.

2.  **Limited Knowledge Base (POC Scope):**
    The current safety logic is hardcoded for a **18-drug prototype** (representing major chronic diseases like Hypertension and Diabetes). It does not contain a comprehensive database of all FDA-approved drugs.

3.  **Out-of-Distribution (OOD) Detection Limitation:**
    The OOD gate (`check_is_prescription`) relies on **semantic validation** of VLM output rather than a dedicated image classifier.
    
    - **Risk**: If the VLM hallucinates medical keywords for a non-prescription image (e.g., a photograph of unrelated objects), the OOD gate may incorrectly classify it as valid input.
    - **Mitigation**: The subsequent **Logic Consistency Check** layer serves as a secondary safety net, flagging anomalous outputs (e.g., impossible dosages, unknown drugs) for human review.
    - **Future Enhancement**: Integrate a lightweight pre-filter (e.g., CLIP-based zero-shot classification) to verify "Is this a medical document?" before VLM inference.

4.  **Language Support:**
    Currently optimized for **Traditional Chinese (Taiwan)** + **English** code-switching. Other languages require fine-tuning.

5.  **Not a Replacement for Pharmacists:**
    This system is a **Clinical Decision Support Tool (CDSS)**, not a licensed medical device. All outputs must be verified by qualified healthcare professionals.
 or FDA/MDR certification.

## ⚙️ Training Data

-   **Dataset:** MedGemma Impact Dataset V5
-   **Size:** 600 Synthetic Images
-   **Composition:** 
    -   70% Standard Prescriptions
    -   30% "Risk Injection" (Adversarial examples with dangerous dosages)
-   **Augmentation:** "Gallery of Horrors" pipeline (Blur, Occlusion, Rotation, Low Light).

## 🛡️ Agentic Architecture

Unlike standard VLMs, this model operates within an **Agentic Loop**:
1.  **Perception:** SigLIP encoder reads the image.
2.  **Reasoning:** LLM generates a safety analysis.
3.  **Self-Correction:** A symbolic logic layer checks for contradictions. If found, the Agent lowers its temperature (0.6 → 0.2) and retries.
4.  **Action:** Outputs structured JSON with a safety status (`PASS` / `WARNING` / `HIGH_RISK`).
