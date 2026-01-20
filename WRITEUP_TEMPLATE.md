# AI Pharmacist Guardian + SilverGuard
**MedGemma Guardian: An Agentic Safety Loop for Elderly Care**

---

## Team
**Yuan-dao Wang** — Full Stack AI Engineer (Solo Project)

## Links
- **Video Demo:** `[Insert YouTube Link]`
- **Code Repository:** `[Insert Kaggle Notebook Link]`
- **Model Weights (Hugging Face):** `[Insert HF Model Link]` (Bonus 1)
- **Live Demo (ZeroGPU):** `[Insert HF Space Link]` (Bonus 2: Interactive A100 App)

---

## Problem Statement

| Statistic | Source |
|-----------|--------|
| **$42B/year** medication error cost | WHO 2024 |
| **7x higher** error rate for patients 65+ | Geriatric Research |
| **53%** of preventable harm at prescribing stage | WHO 2024 |

**Challenge:** Elderly patients cannot read complex prescription labels → dangerous medication errors.
**Strategic Context:** We use Taiwan as a **High-Complexity Stress Test** (Code-Switching + Super-Aged Society) to proxy future global challenges.

---

## Solution: MedGemma Agentic Workflow

**AI Pharmacist Guardian** deploys **MedGemma 1.5-4B** as a self-correcting agent:

```
Input → VLM Reasoning → Logic Check → [RETRY if failed] → Confidence → Output/Human Flag
```

### Key Agentic Features
- **Self-Correction Loop**: On logic failure, agent modifies prompt + retries with lower temperature
- **Input Gate**: Rejects blurry/OOD images before processing
- **Human-in-the-Loop**: Confidence <80% → "Human Review Needed"

### SilverGuard (Social Impact)
- 🗣️ TTS voice readout for visually impaired
- 📅 Large-font calendar (28px+) for cognitively impaired
- 🎨 High-contrast colors (WCAG 2.1 AAA)

---

## Technical Details

| Parameter | Value |
|-----------|-------|
| Model | `google/medgemma-1.5-4b-it` |
| Method | QLoRA (4-bit nf4), LoRA r=16 |
| Training | 600 synthetic images, 3 epochs, ~54 min on T4 |
| Risk Injection | 30% adversarial (elderly overdose, wrong timing, drug interactions, renal risk) |

### Confidence Formula
`C = α × P_mean + (1-α) × P_min` where α=0.7  
*Rationale: Amplify uncertain tokens → prefer Human Review over missed errors*

### Metrics (Test N=60)
| Metric | Result |
|--------|--------|
| High Risk Recall | ~95%+ |
| Accuracy | ~93%+ |
| Human Review Rate | ~5-10% |

> **Sim2Real Note:** Trained on synthetic data as POC. Input Gate mitigates real-world noise by rejecting low-quality images.

### 🛡️ The "Fail-Safe" Architecture

Unlike standard LLMs that hallucinate an answer 100% of the time, **MedGemma Guardian is architected to REJECT uncertainty**:

| Condition | System Behavior | Rationale |
|-----------|-----------------|-----------|
| **Confidence < 80%** | Triggers `HUMAN_REVIEW` flag | Uncertain predictions are dangerous |
| **Logic Check Fails** | Agent retries 2× with modified prompt | Self-correction attempt |
| **Still Inconsistent** | Aborts and escalates to pharmacist | Safety over availability |

> **Design Philosophy:** *"We would rather provide NO answer than a WRONG answer."*  
> This trades **Availability** for **Safety** — the only acceptable trade-off in medical AI.

---

## Impact & Deployment

- **Cost**: <$0.001/diagnosis on T4 GPU
- **Privacy**: 100% local PHI processing (no patient data leaves device)
- **Hybrid TTS**: gTTS (online, best quality) → pyttsx3 (offline fallback) → Visual-only (always works)
- **Edge Target**: NVIDIA Jetson Orin Nano (67 TOPS, 15W)

> **Deployment Note:** Core model inference is fully offline-capable. SilverGuard TTS degrades gracefully from cloud (high quality) to local (robotic) to visual-only, ensuring the system remains functional in intermittent connectivity environments typical of rural clinics.

**Projected Savings:** 1 pharmacy × 10K prescriptions/month → ~$10-20K/month prevented harm

---

## Responsible AI

- ⚠️ **Not SaMD**: Research prototype, requires pharmacist oversight
- ⚖️ **Sim2Real Gap**: Model trained on **synthetic images only** (perfect fonts/layouts). Real-world degradation expected → Human-in-the-Loop mitigation
- 📚 **Prototype Scope**: Only **12 drugs** in knowledge base. Unknown drugs → "NOT_FOUND" → Human review (safety architecture > knowledge breadth)
- 🛡️ **Fail-Safe**: "When in doubt, fail safely" — refuse rather than guess

---

## Design Philosophy: Why This Architecture?

### 1. Deterministic Guardrailing > Open-Ended Autonomy
> *"For a creative writing bot, open-ended tool use is great. For a pharmacist potentially overdosing an 88-year-old, 'creative autonomy' is a bug, not a feature."*

We deliberately constrained the Agent within a structured `retry-loop` because **Reliability > Novelty** in life-critical systems. The `MAX_RETRIES=2` limit is a **latency-aware safety boundary** for edge deployment.

### 2. Deterministic Guardrails (Not Magic, Just Engineering)
> *"Why use RegEx for dosage checks? Because LLMs cannot do arithmetic reliably."*

We intentionally combine **Neural perception** (MedGemma for reading and understanding) with **Symbolic validation** (Python regex and rule-based checks). This is not a novel research breakthrough—it's **pragmatic fail-safe engineering**:

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Perception** | MedGemma VLM | Extract text, understand context |
| **Validation** | Python Regex | Verify dosage format (e.g., `\d+mg`) |
| **Logic** | Rule-based checks | Flag impossible values (age > 120) |

> **Honest Limitation:** The symbolic layer's intelligence is bounded by our handwritten rules. It catches obvious errors (missing units, impossible ages) but cannot detect subtle clinical contraindications not encoded in our rules.

*References:*
- arXiv 2024: *"Neuro-Symbolic AI in healthcare...integrates symbolic reasoning with neural networks for more robust, explainable, and reliable AI systems."*
- GSM-Symbolic (Apple, 2024): *"LLMs exhibit noticeable variance in arithmetic responses...performance declines when numerical values are changed, even if underlying logic remains the same."*

### 3. Synthetic Data as Intentional POC Strategy
> *"Only synthetic data allows us to inject rare, lethal edge cases (e.g., renal failure drug interactions) that are statistically impossible to collect ethically from real hospital logs."*

This project validates the **logic pipeline**. Sim2Real transfer is the next phase, not a forgotten step.

---

## Known Limitations

| Limitation | Design Rationale |
|------------|------------------|
| **Regex-based Dose Validation** | Conservative fail-safe: Flags malformed doses (e.g., missing units) for human review rather than guessing. |
| **gTTS Cloud Dependency** | Hybrid Privacy Architecture: PHI (patient data, images) stays local; only non-sensitive synthesized text uses TTS API. |
| **MAX_RETRIES = 2** | Latency trade-off: Limits edge-deployment inference time while maintaining Agentic self-correction. |
| **Synthetic Training Data** | Privacy-by-design: No real patient data used. Model robustness validated via augmentation (blur, noise). |

---

## 🌍 Future Roadmap: Scaling SilverGuard Globally

| Phase | Target Market | Objective | Timeline |
|-------|---------------|-----------|----------|
| **Phase 1** | 🇹🇼 Taiwan | Deploy in 5 community pharmacies to stress-test mixed-script (EN/ZH) parsing | Q2 2026 |
| **Phase 2** | 🇺🇸 US Hispanic | Swap output layer to **Spanish (Español)** to serve **60M+ Hispanic population** facing similar language barriers in healthcare | Q4 2026 |
| **Phase 3** | 🌍 Global South | Port quantized model to **Android (Google MediaPipe)** for offline use by home care nurses in **rural Africa** (no internet required) | 2027 |
| **Phase 4** | 🏥 Enterprise | Replace hardcoded `DRUG_DATABASE` with **RAG (Retrieval-Augmented Generation)** connecting to **RxNorm/Micromedex** APIs for real-time drug interaction checking | 2027 |
| **Phase 5** | 🔌 Air-Gapped | Integrate **Coqui TTS** or **pyttsx3** for 100% offline voice synthesis, enabling deployment in network-isolated environments | 2028 |
| **Phase 6** | 🔐 Federated Learning | Enable **privacy-preserving model updates** across hospitals. Local gradients aggregated globally—**PHI never leaves premises**. (Google AI 2016 technique) | 2028+ |

> **Market Expansion Logic:** Taiwan is the *highest-complexity stress test* (code-switching + super-aged society).  
> Once validated, the same architecture scales to any language pair with minimal adaptation.
> 
> **Scalability Acknowledgment:** Current `DRUG_DATABASE` contains 12 drug types—sufficient for POC validation but not production. Phase 4 addresses this limitation.

---

## Citation

```
Yuan-dao Wang. AI Pharmacist Guardian: The MedGemma Impact Challenge.
https://kaggle.com/competitions/med-gemma-impact-challenge, 2026. Kaggle.
```
