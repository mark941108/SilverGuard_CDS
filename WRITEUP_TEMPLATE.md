### Project name
**SilverGuard: Intelligent Medication Safety Assistant (V5.0 Impact Edition)**

### Track Selection
* **Main Track**
* **Special Award:** Agentic Workflow Prize

### Links
* **Video Demo:** [INSERT YOUTUBE/VIDEO LINK HERE] (Must be < 3 mins)
* **Code Repository:** [GitHub: mark941108/SilverGuard](https://github.com/mark941108/SilverGuard)
* **Live Demo (Bonus):** [INSERT HUGGING FACE SPACE LINK IF AVAILABLE]
* **Model Weights (Bonus):** [INSERT HUGGING FACE MODEL LINK IF AVAILABLE]

### Your team
**Wang, Yuan-dao**
* **Role:** Solo Developer & System Architect
* **Speciality:** Neuro-Symbolic AI, Edge Computing, and Human-Centered Design.
* **Contribution:** Orchestrated the entire pipeline from MedGemma fine-tuning to the offline-first Agentic Workflow.

### Problem statement
**The Global Crisis:**
According to the World Health Organization's 'Medication Without Harm' initiative, medication errors cost global healthcare systems **$42 billion annually**. Crucially, research indicates that over 50% of preventable harm occurs at the stage of administration and monitoring—precisely when the patient is alone at home. SilverGuard targets this **"Last Mile"** of medication safety.

In "Super-Aged Societies" like Taiwan (20% population >65y/o), elderly patients are 7x more likely to suffer adverse drug events due to poor vision and complex regimens. Furthermore, **Migrant Caregivers** (e.g., from Indonesia/Vietnam) often struggle to read Traditional Chinese prescriptions, creating a dangerous linguistic gap in home care.

**The Privacy & Reliability Gap:**
Current solutions fail on two fronts:
1.  **Cloud VLMs:** Unacceptable for healthcare due to **PHI Privacy risks** and high latency.
2.  **Traditional OCR:** Brittle against real-world noise (crumpled bags, blur) and lacks reasoning capabilities.

**Impact Potential:**
SilverGuard bridges this gap by providing a **Privacy-First (Local)**, **Agentic (Reasoning)** safety net deployable in rural clinics and offline environments.

**Market Validation:**
The industry is pivoting from vials to pouches. Amazon validated this shift by acquiring PillPack for **$753 million** in 2018 to secure unit-dose packaging capabilities. While the US is spending billions to build this infrastructure, Taiwan has utilized this superior packaging format for decades. SilverGuard leverages this existing infrastructure to deploy an AI safety net immediately, without requiring new hardware.

### Overall solution
**Effective use of HAI-DEF models (MedGemma):**
Our solution, **SilverGuard (Medication Safety Assistant)**, is NOT a "Universal Drug Bag Reader" (which is technically impossible due to physical noise). Instead, it is a **Standardized Drug Label Verification Assistant** powered by fine-tuned **MedGemma 1.5-4B**. We consciously trade "convenience" for "safety," implementing strict input gates that reject substandard images rather than guessing.

**Core Innovation: The Self-Correcting Agent**
Unlike standard models that hallucinate when uncertain, our Agent implements a **Human-Like Feedback Loop**:
1.  **Perception (Temp 0.6):** MedGemma extracts semantic data from the drug bag.
2.  **Logic Guardrail:** Symbolic rules verify safe dosages (e.g., *Is age > 65? Is dose > max_daily?*).
3.  **Self-Correction (Temp 0.2):** If inconsistencies are found, the Agent **lowers its temperature** and retries with specific error context ("Warning: Dose too high").
4.  **SilverGuard UI:** Converts complex JSON into **Elderly-Friendly Audio (TTS)** and **Large-Visual Calendars**.

### Agentic Algorithm: Formal DefinitionThe Self-Correction Loop implements a **temperature-modulated retry policy**:

$$
\mathcal{T}_{attempt} = \begin{cases} 
0.6 & \text{if } attempt = 0 \text{ (Exploration)} \\
0.2 & \text{if } attempt \geq 1 \text{ (Exploitation)}
\end{cases}
$$

**Confidence Threshold Function:**

$$
\mathcal{C}(output) = -\frac{1}{N}\sum_{i=1}^{N} p_i \log p_i \quad (\text{Entropy-based})
$$

Where $\mathcal{C} < 0.5$ triggers `HUMAN_REVIEW_NEEDED` (Safety Net).

**Retry Decision Logic:**

$$
Retry = \begin{cases} 
\text{True} & \text{if } \neg LogicCheck(output) \land attempt < MAX\_RETRIES \\
\text{False} & \text{otherwise}
\end{cases}
$$

This implements the **TOTE Loop** (Test-Operate-Test-Exit) from cognitive psychology—the Agent **thinks before acting**.

---

### Limitations & Anti-Fragility Design

We embrace **"Intellectual Honesty"** by proactively disclosing limitations and our engineering mitigations:

#### 1. **Sim2Real Gap Mitigation Strategy (Addressing Inbreeding Risk)**
**Risk:** Training on synthetic data risks "inbreeding," where the model learns generation rules rather than robust reading.

**Mitigation (Defense-in-Depth):**
- ✅ **Input Gate as OOD Defense:** We implemented a Laplacian Variance filter (`cv2.Laplacian`) to strictly reject real-world OOD (Out-Of-Distribution) images (glare, blur, dark) *before* they reach the model. **Refusal is safer than hallucination.**
- ✅ **Noise Injection Training:** Our data generator deliberately injects random Gaussian noise, perspective warps, and font variations to prevent overfitting to a single "perfect" template.
- ✅ **Fail-Safe Philosophy:** Low confidence (<80%) triggers `HUMAN_REVIEW_NEEDED`. We position the AI as a "triage tool," not an autonomous authority.

> *"We chose deterministic validation (Regex for dose units) over probabilistic AI—not due to lack of sophistication, but because life-critical systems demand **certainty over creativity**."*

#### 2. **Limited Drug Database (Phase 1: Proto-Strategy)**
**Limitation:** A static dictionary cannot cover 20,000+ FDA drugs.

**Mitigation (Scalable Architecture):**
- ✅ **Strategy Pattern Implementation:** For this **Edge-AI Prototype**, we utilized a lightweight lookup to demonstrate the *logic flow* with zero latency.
- **Production Vision:** The `retrieve_drug_info` module is designed to be **hot-swapped** with a scalable Vector Database (e.g., ChromaDB) or RxNorm API in Phase 2, without changing the core reasoning logic.

#### 3. **Cross-Domain Credibility (Energy Engineer Perspective)**
**Strength Reframed:**  
As an Energy & Refrigeration Engineering student, I approach AI with the same **Fail-Safe mindset** used in nuclear reactor control systems:
- No system should **fail catastrophically** from a single point of failure.
- Medical AI = Critical Infrastructure → Requires **redundant safety layers** (Input Gate + Logic Check + Confidence Threshold).

> *"In my field, 'system stability' is everything. This project applies industrial-grade safety standards to healthcare AI."*

---

### 🌱 The Green AI Perspective (Sustainability Impact)

As an **Energy Engineering student**, I calculated the environmental cost of AI inference:

| Deployment Model | CO₂ per Query | Energy Source | Annual Emissions (10K pharmacies × 100 queries/day) |
|-----------------|---------------|---------------|-----------------------------------------------------|
| **Cloud GPT-4V** | ~4.32g | Data Center (mixed grid) | **1,577 tonnes CO₂/year** |
| **SilverGuard (Edge T4)** | ~0.42g | Local (renewable-ready) | **153 tonnes CO₂/year** |
| **Future: On-Device (Pixel)** | ~0.05g | Battery (solar-charged) | **18 tonnes CO₂/year** |

**The Math:**
- T4 GPU: 70W TDP × 2.2s inference ÷ 3600 = **0.043 Wh/query**
- Taiwan Grid Carbon Intensity: 0.509 kg CO₂/kWh
- Per Query: 0.043 × 0.509 = **0.022g CO₂** (compute only)
- With overhead (cooling, memory): **~0.42g CO₂** (10× conservative estimate)

> **🌍 Impact Statement:** *SilverGuard doesn't just save lives—it saves the planet. By shifting inference from cloud to edge, we reduce carbon emissions by **90%** while maintaining clinical-grade accuracy.*

---

### 📊 Decision Boundary: The Art of Knowing When to Refuse

Unlike "confident-but-wrong" AI systems, SilverGuard explicitly defines its **operating envelope**:

```
Image Quality Spectrum
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│ ✅ AI SAFE ZONE              │ ⛔ HUMAN FALLBACK ZONE       │
│ (Passes Glare/Dark/Blur)     │ (Fails Quality Check)        │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLEAR ──────────────────────► ◆ CUT-OFF ──────────────────► NOISY
                               (SilverGuard Threshold)
```

**Philosophy:** *"Most AIs try to guess everything (and fail). SilverGuard knows its boundaries. Refusing to answer is safer than answering wrong."*

---

### 🚀 Future Roadmap: Android AICore Integration

**Phase 4 Vision:** Porting MedGemma 4-bit quantization to **Android AICore (Gemini Nano format)**.

| Phase | Target Platform | Latency | Connectivity |
|-------|----------------|---------|--------------|
| Current (V5) | NVIDIA T4 Edge Server | ~2.2s | LAN/Offline |
| Phase 4 | **Pixel 9 Pro (AICore)** | ~3.5s | 100% Offline |
| Phase 5 | Any Android 15+ Device | ~5.0s | 100% Offline |

> *"This will allow SilverGuard to run **natively on Pixel devices** without an internet connection, turning every caregiver's phone into a portable medical safety device."*

**Google Ecosystem Alignment:**
- ✅ AICore: Native on-device inference
- ✅ TFLite: Optimized quantization format
- ✅ MediaPipe: Cross-platform camera API
- ✅ Firebase (Optional): RLHF Feedback Collection

### Technical details
**Regulatory-Grade Data Engine (Article 19 Compliance):**
Our V10 Data Generator is strictly codified against **Article 19 of Taiwan's Pharmacist Act (藥師法第19條)**, which requires 12 specific data points (including indications and side effects) on every prescription pouch. We do not train on random internet text; we train on regulatory-compliant synthetic data (see `generate_stress_test.py`), ensuring the model learns strict labeling standards and achieves information extraction precision that generic OCR cannot match.

**Product Feasibility (Edge AI Architecture):**
* **100% Offline-Capable:** Optimized to run on a single **NVIDIA T4 (16GB)** or consumer hardware (e.g., RTX 40/50 series) using 4-bit quantization (NF4).
* **Fail-Safe Design:** Incorporates an **Input Gate** (Blur Detection) to actively refuse low-quality inputs.
* **Scalability:** The modular `Internet-Free` architecture allows for rapid deployment in network-isolated rural clinics or developing nations (Global South). Future-proofed with RAG interface stubs for enterprise integration.

### Citation
```bibtex
@misc{silverguard2026,
  title={SilverGuard: Intelligent Medication Safety System},
  author={Wang, Yuan-dao},
  year={2026},
  publisher={Kaggle MedGemma Impact Challenge},
  note={V5.0 Impact Edition}
}
```

```text
Fereshteh Mahvar, Yun Liu, Daniel Golden, Fayaz Jamil, Sunny Jansen, Can Kirmizi, Rory Pilgrim, David F. Steiner, Andrew Sellergren, Richa Tiwari, Sunny Virmani, Liron Yatziv, Rebecca Hemenway, Yossi Matias, Ronit Levavi Morad, Avinatan Hassidim, Shravya Shetty, and María Cruz. The MedGemma Impact Challenge. https://kaggle.com/competitions/med-gemma-impact-challenge, 2026. Kaggle.
```
