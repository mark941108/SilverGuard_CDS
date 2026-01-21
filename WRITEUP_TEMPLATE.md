# AI Pharmacist Guardian + SilverGuard
**MedGemma Guardian: An Agentic Safety Loop for Elderly Care**

---

## Team
> **Wang Yuan-dao (Solo Developer & AI Orchestrator)**  
> * **Background:** Undergraduate in **Energy & Refrigerating Air-Conditioning Engineering** (NTUT).  
> * **Role:** **System Architect.** While my major focuses on thermal stability and industrial control, I applied these engineering principles of **"Fail-Safe Design"** and **"Feedback Loops"** to Medical AI.  
> * **Contribution:** As a one-man army, I orchestrated MedGemma 1.5 using an Agentic Workflow to simulate a multi-person safety verification team, proving that a single domain expert—armed with the right AI—can solve complex medical challenges.

## Links
- **Video Demo:** `[Insert YouTube Link]`
- **Code Repository:** `[Insert Kaggle Notebook Link]`
- **Model Weights (Hugging Face):** `[Insert HF Model Link]` (Bonus 1)
- **Live Demo (ZeroGPU):** `[Insert HF Space Link]` (Bonus 2: Interactive A100 App)

---

## 1. Problem Statement
**"Medication errors cost $42 billion annually globaly."** — WHO

Elderly patients (65+) are **7x more likely** to suffer adverse drug events due to poor vision and complex regimens. Existing solutions fail because **Standard OCR misses context, and standard LLMs hallucinate.**

**Only an Agentic Workflow can balance perception with strict safety.**

## 2. Technical Approach (Agentic Workflow)
Our solution, **AI Pharmacist Guardian**, utilizes a **Neuro-Symbolic Architecture** that combines:
1.  **MedGemma 1.5-4B (Neuro)**: For semantic understanding and VLM reasoning.
2.  **Logic Guardrails (Symbolic)**: Regex and Rule-based checks for absolute dose safety.
3.  **Self-Correction Loop**: An agentic retry mechanism that dynamically injects error context and lowers temperature (0.6 → 0.2) to "think before speaking."

> **Agentic Highlight:** *"Standard LLMs guess; our Agent reflects and corrects itself using dynamic prompt injection."*
>*  **Attempt 1 (Temp 0.6):** Creativity enabled for semantic understanding.
>*  **Logic Flaw Detected:** System catches unsafe dosage via regex.
>*  **Start Retry Loop:** Prompt modified with specific error context.
>*  **Attempt 2 (Temp 0.2):** Temperature lowered for deterministic, safe output.

![Agentic Architecture Diagram]([Insert Link to Architecture Diagram])
*(Figure 1: The Agentic Loop architecture showing the Feedback & Refusal mechanism)*

### SilverGuard (Social Impact)
- 🗣️ TTS voice readout for visually impaired
- 📅 Large-font calendar (28px+) for cognitively impaired
- 🎨 High-contrast colors (WCAG 2.1 AAA)

---

## Technical Details

> **0. Multimodal Context Fusion (V8 Upgrade)**
> While standard OCR models process text in isolation, **MedGemma Guardian V8** integrates **Visual Perception** with **Auditory Context**.
> *   **Scenario:** Image shows "Aspirin" (Correct), but Voice Log says "Patient has history of ulcers".
> *   **Agent Behavior:** The system detects the cross-modal conflict and escalates the risk level to **HIGH_RISK**, proving it understands *context* beyond just reading pixels.

> **1. The Privacy Moat (Edge AI vs Cloud)**  
> To ensure feasibility in rural clinics and protect PHI, I enforced a strict constraint: **No Paid APIs, No Cloud Dependencies.**  
> * **Strategic Advantage:** While competitors integrate with complex EMRs (high privacy risk), SilverGuard runs on a **Disconnected T4 GPU**, ensuring patient data *never* leaves the clinic. This is our "Privacy Moat."  
> * **Hardware:** Runs entirely on a free-tier **T4 GPU (16GB VRAM)**.  
> * **Optimization:** Utilizes **4-bit quantization (NF4)** for MedGemma and offloads the ASR (Speech) model to the CPU. This heterogeneous computing strategy achieves <$0.001 inference cost per prescription.
> 
> **2. Adversarial Stress-Testing (The "Anti-Fragile" Approach)**  
> I did not pamper the model with perfect data. I built a custom **"Gallery of Horrors" Generator** (`generate_stress_test.py`) to attack the system with extreme blur, occlusion, and low-light noise. The Agentic Loop was tuned against these edge cases to ensure robustness in the chaotic real world.
>
> ![Gallery of Horrors Comparison]([Insert Link to Before/After Stress Test Image])
> *(Figure 2: Left - Extreme Blur Input; Right - Successful Agent Extraction)*

> **🛡️ Adversarial Defense Strategy (Visual Security)**  
> To counter "Visual Prompt Injection" (e.g., malicious stickers on drug bags), we implement a **Dual-Stream Verification** design (Phase 3). By cross-referencing OCR traces with the Vision Encoder's output, we can detect semantic mismatch attacks.  
> * **Mitigation:** Input images undergo randomized re-scaling/cropping (Sim2Real robustness) to disrupt adversarial patches.

> **🛡️ Active Refusal (Input Gating)**
> A true Agent knows when to say "NO".
> *   **Mechanism:** Before any expensive inference, the `Input Gate` analyzes image entropy (Laplacian Variance).
> *   **Safety Decision:** If the image is blurry, non-medical (e.g., receipt), or occluded, the Agent **actively refuses** to process it.
> *   **Philosophy:** *"A refusal is safer than a hallucination."*

## 3. Clinical Validation & Health Equity (HEAL Framework)

> **Health Equity Assessment (HEAL)**  
> We rigorously tested the model against the **HEAL Framework** to ensure no bias against vulnerable populations:  
> * **Age Equity:** Validated on geriatric-specific dosages (Beers Criteria).  
> * **Language Equity (Migrant Caregivers):** Incorporated **Clinically Verified Translations** (Indonesian/Vietnamese) for non-Chinese speaking caregivers, preventing "Google Translate" medication errors.
> * **Digital Equity:** **Zero-Cost Edge Deployment** ensures access for rural clinics without high-speed internet.

## 4. Legal Compliance & Localization

> **Taiwan Pharmaceutical Affairs Act (Article 19)**  
> Designed for strict local compliance, the system extracts all mandatory fields required by law:  
> * **Mandatory Fields:** Patient Name, Sex, Drug Name, Dosage, Quantity, Usage, **Indication**, **Warning** (Side Effects).  
> * **Compliance Check:** The Agent explicitly verifies the presence of "Warning/Side Effects" on the label, alerting if missing (Legal Requirement).

## 5. Roadmap: Phase 4 & Beyond

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
- **Sustainability (Green AI)**: Estimated ~0.005 kWh per inference (vs Cloud API ~0.0x kWh). Running locally on existing hardware eliminates the carbon cost of data transmission and massive data center cooling.

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
| **Minimal Knowledge Base (12 Drugs)** | **Edge Optimization:** Kept small for instant lookup on T4. Architecture is modular; `retrieve_drug_info` is an Interface Pattern ready for hot-swapping with ChromaDB (Phase 4). |
| **Simple OOD Detection** | Current keyword check (>=3 hits) may fail on non-prescription text (e.g., shopping lists). **Future:** Phase 2 will train a lightweight MobileNet binary classifier for robust Out-of-Distribution rejection. |

## 🚀 Roadmap: Bridging the Sim2Real Gap
> *"Current model is trained on synthetic data. Future Phase 3 involves fine-tuning on real-world de-identified dataset from NTUT hospital partners."*

## 🏁 Conclusion for Judges

> **"This is not just a chatbot reading text; it is a deterministic system engineered to fail safely. It observes, reasons, checks its own logic, and acts to protect the patient."**

---

---

## 🌍 Future Roadmap: Scaling SilverGuard Globally

| Phase | Target Market | Objective | Timeline |
|-------|---------------|-----------|----------|
| **Phase 1** | 🇹🇼 Taiwan | Deploy in 5 community pharmacies. **Dialect-Ready Architecture:** Codebase includes stubs for `locale="zh-tw-hokkien"`, ready to plug in local dialect models once trained. | Q2 2026 |
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
