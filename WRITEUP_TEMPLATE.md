### Project name
**SilverGuard: AI Pharmacist Guardian (V5.0 Impact Edition)**

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
According to the WHO, **medication errors cost $42 billion annually**. In "Super-Aged Societies" like Taiwan (20% population >65y/o), elderly patients are 7x more likely to suffer adverse drug events due to poor vision and complex regimens.

**The Privacy & Reliability Gap:**
Current solutions fail on two fronts:
1.  **Cloud VLMs:** Unacceptable for healthcare due to **PHI Privacy risks** and high latency.
2.  **Traditional OCR:** Brittle against real-world noise (crumpled bags, blur) and lacks reasoning capabilities.

**Impact Potential:**
SilverGuard bridges this gap by providing a **Privacy-First (Local)**, **Agentic (Reasoning)** safety net deployable in rural clinics and offline environments.

### Overall solution
**Effective use of HAI-DEF models (MedGemma):**
Our solution, **AI Pharmacist Guardian**, is a **Neuro-Symbolic Agentic Workflow** powered by a fine-tuned MedGemma 2 (2B/9B). It acts not just as a reader, but as a meticulous pharmacist.

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

#### 1. **Synthetic Data (Sim2Real Gap)**
**Limitation:** Model trained exclusively on programmatically generated drug bags.

**Mitigation (Anti-Fragility):**
- ✅ **"Gallery of Horrors" Stress Test:** We deliberately attack our model with 10 extreme edge cases (blur, occlusion, water damage).
- ✅ **Input Gate (Laplacian Variance):** Rejects blurry images pre-inference. **Refusal is safer than hallucination.**
- ✅ **Fail-Safe Philosophy:** When uncertain → `HUMAN_REVIEW_NEEDED` (not a failure, a feature).

> *"We chose deterministic validation (Regex for dose units) over probabilistic AI—not due to lack of sophistication, but because life-critical systems demand **certainty over creativity**."*

#### 2. **Limited Drug Database (12 Drugs POC)**
**Limitation:** Current knowledge base covers only 12 high-risk chronic disease medications.

**Mitigation (Modular Architecture):**
- ✅ **Decoupled Design:** The `retrieve_drug_info()` function serves as a **RAG Interface Stub**. Replacing the local dictionary with RxNorm/Micromedex API requires only 5 lines of code (see `AI_Pharmacist_Guardian_V5.py` line 363).
- ✅ **Graceful Degradation:** Unknown drugs trigger `UNKNOWN_DRUG` status → Manual Review (prevents hallucination).

> *"This is a **POC (Proof of Concept)** demonstrating safety architecture, not a production drug encyclopedia. The modular design allows scaling to 20,000+ FDA drugs without retraining the model."*

#### 3. **Cross-Domain Credibility (Energy Engineer Perspective)**
**Strength Reframed:**  
As an Energy & Refrigeration Engineering student, I approach AI with the same **Fail-Safe mindset** used in nuclear reactor control systems:
- No system should **fail catastrophically** from a single point of failure.
- Medical AI = Critical Infrastructure → Requires **redundant safety layers** (Input Gate + Logic Check + Confidence Threshold).

> *"In my field, 'system stability' is everything. This project applies industrial-grade safety standards to healthcare AI."*

### Technical details
**Product Feasibility (Edge AI Architecture):**
* **100% Offline-Capable:** Optimized to run on a single **NVIDIA T4 (16GB)** or consumer hardware (e.g., RTX 40/50 series) using 4-bit quantization (NF4).
* **Fail-Safe Design:** Incorporates an **Input Gate** (Blur Detection) to actively refuse low-quality inputs.
* **Scalability:** The modular `Internet-Free` architecture allows for rapid deployment in network-isolated rural clinics or developing nations (Global South). Future-proofed with RAG interface stubs for enterprise integration.

### Citation
```bibtex
@misc{silverguard2026,
  title={SilverGuard: AI Pharmacist Guardian},
  author={Wang, Yuan-dao},
  year={2026},
  publisher={Kaggle MedGemma Impact Challenge},
  note={V5.0 Impact Edition}
}
```

```text
Fereshteh Mahvar, Yun Liu, Daniel Golden, Fayaz Jamil, Sunny Jansen, Can Kirmizi, Rory Pilgrim, David F. Steiner, Andrew Sellergren, Richa Tiwari, Sunny Virmani, Liron Yatziv, Rebecca Hemenway, Yossi Matias, Ronit Levavi Morad, Avinatan Hassidim, Shravya Shetty, and María Cruz. The MedGemma Impact Challenge. https://kaggle.com/competitions/med-gemma-impact-challenge, 2026. Kaggle.
```
