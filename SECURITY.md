# Security Policy

## 🛡️ The "Privacy Moat" Architecture

We take patient privacy seriously. This project is architected as an **Edge AI** solution to ensure compliance with HIPAA and GDPR principles by design.

### Data Privacy & Egress Policy

*   **100% Local Processing:** All Patient Health Information (PHI), including drug bag images and voice logs, is processed locally on the deployment device (e.g., T4 GPU, Jetson Orin).
*   **Zero Data Egress:** No PHI is transmitted to external cloud servers or third-party APIs for inference.
*   **Ephemeral Storage:** RAM is cleared after each inference session. We do not persist patient data on the device storage unless explicitly configured for local logging (which is disabled by default).

> **Verification:** You can verify this behavior by running the application in an offline environment (air-gapped). The core inference pipeline remains fully functional.

## 🐛 Reporting a Vulnerability

As this is a medical safety project, we welcome security research, particularly regarding:
*   **Adversarial Attacks:** Visual prompt injections (e.g., malicious stickers on drug bags).
*   **Model Extraction:** Attempts to recover training data (though we use synthetic data, we value robustness).

If you discover a vulnerability, please do **NOT** open a public issue. Instead, please report it via:

*   **Email:** [YOUR_EMAIL@EXAMPLE.COM] (Replace with actual contact if applicable or "Kaggle Message")
*   **Subject:** `[SECURITY] - MedGemma Guardian Vulnerability`

We will acknowledge receipt within 48 hours.

## 📦 Dependency Security

To mitigate supply chain attacks and ensure reproducibility, we pin our critical dependencies to secure versions.

| Package | Version | Rationale |
| :--- | :--- | :--- |
| `transformers` | `>=4.50.0` | Contains critical security patches for model loading |
| `safetensors` | `Required` | We prioritize `.safetensors` weights over `.bin` (pickle) to prevent arbitrary code execution |
| `gradio` | `Latest` | For secure UI rendering (if applicable) |

## ⚠️ Medical Safety Disclaimer

This software is a **Clinical Decision Support (CDS) Prototype** and is **NOT** a substitute for professional medical advice, diagnosis, or treatment. It is not an FDA-cleared medical device.
