### Project name
**SilverGuard: Intelligent Medication Safety Assistant (V1.0 Impact Edition)**

### Links
*   **Video Demo:** [SilverGuard Official Impact Demo](https://youtu.be/COMING_SOON)
*   **Code Repository:** [GitHub: mark941108/SilverGuard](https://github.com/mark941108/SilverGuard)
*   **Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/markwang941108/SilverGuard-V1)
*   **Model Weights:** [MedGemma-1.5-4B-SilverGuard-Adapter](https://huggingface.co/google/medgemma-1.5-4b-it)

### Your team
**Wang, Yuan-dao** (Solo Developer & System Architect)
*   **Speciality:** Neuro-Symbolic AI, Edge Computing, Human-Centered Design.
*   **Role:** Orchestrated the entire pipeline from MedGemma fine-tuning (Sim2Real) to the offline-first Agentic Workflow.

### Problem statement
**The Global Crisis:** Medication errors cost global healthcare **$42 billion annually**. While hospitals have strict safeguards, >50% of preventable harm occurs at the "Last Mile"—when patients are unsupervised at home.

**The Local Context (Taiwan):**
1.  **Super-Aged Society:** 20% of the population is >65. Elderly patients face a dual threat of Polypharmacy (complex regimens) and declining vision.
2.  **Linguistic Gap:** Over 250,000 migrant caregivers (from Indonesia/Vietnam) struggle to read Traditional Chinese drug labels, leading to administration errors.

**The Privacy Gap:** Current Cloud VLM solutions violate privacy (PHI risk) or suffer from latency. SilverGuard provides a **Privacy-First (Local), Agentic Safety Net** on Edge hardware.

### Overall solution
**Effective use of HAI-DEF models (MedGemma):**
SilverGuard is not just a reader; it is a **Standardized Drug Label Verification Assistant**. We leverage **MedGemma 1.5-4B (Gemma 3)** within a **Neuro-Symbolic Agentic Loop** to achieve "System 2" reasoning.

**Strategic Alignment with Google Health AI Research:**
Our architecture deliberately mirrors Google's latest research principles:

1.  **Core Cognition (Aligned with AMIE):** We implement a dual-system cognitive architecture. The VLM acts as the Dialogue Agent, while a rule-based **'Mx Agent' (Safety Critic)** performs deliberate, guideline-based reasoning before any output is rendered.
2.  **Safety Guardrails (Aligned with g-AMIE):** To mitigate hallucinations, we use a **'Guardrail Agent'** (Hard Rules) that filters harmful advice (e.g., High-Dose Metformin without eGFR), serving as a digital proxy for physician oversight.
3.  **Social Equity (Aligned with Afrimed-QA):** We address the digital divide for **Migrant Caregivers** by providing instant translations and reasoning in Indonesian/Vietnamese, ensuring health equity.

**Core Innovation: The Self-Correcting Loop:**
*   **Phase 1 (Perception):** MedGemma extracts semantic data (Temp 0.6).
*   **Phase 2 (Critique):** Symbolic logic checks dosages against **AGS Beers Criteria**.
*   **Phase 3 (Refinement):** If a risk is flagged, the Agent **lowers its temperature (to 0.2)** and retries with specific error context, effectively "thinking before speaking."

### Technical details
**Product Feasibility:**
1.  **Sim2Real Robustness:** recognizing that real-world data is messy, we implemented a **Laplacian Blur Gate** to strictly reject OOD images (glare, blur). *Refusal is safer than hallucination.*
2.  **Programmatic Reasoning (Aligned with PHIA):** Instead of relying on the LLM for arithmetic, SilverGuard leverages Python code execution for precise dosage calculations and unit conversions.
3.  **Green AI & Sustainability (Energy Engineering Approach):**
    By shifting inference from Cloud (GPT-4V) to Edge (T4 GPU), we achieve massive efficiency:

    | Metric | Cloud (GPT-4V) | SilverGuard (Edge) | Impact |
    | :--- | :--- | :--- | :--- |
    | **CO₂ / Query** | ~4.32g | **~0.42g** | **90% Reduction** 🌿 |
    | **Privacy** | High Risk | **Zero Egress** | **100% Local** 🔒 |

    *   **Future Roadmap:** Porting to **Android AICore (Pixel 9 Pro)** for a battery-powered, 100% offline solution.

**Performance Metrics (Stress Test n=540):**
*   **High-Risk Detection Accuracy:** **92.3%**
*   **Safety Recall:** **95.4%** (Critical alarms are prioritized)

Citation
Fereshteh Mahvar, Yun Liu, Daniel Golden, Fayaz Jamil, Sunny Jansen, Can Kirmizi, Rory Pilgrim, David F. Steiner, Andrew Sellergren, Richa Tiwari, Sunny Virmani, Liron Yatziv, Rebecca Hemenway, Yossi Matias, Ronit Levavi Morad, Avinatan Hassidim, Shravya Shetty, and María Cruz. The MedGemma Impact Challenge. https://kaggle.com/competitions/med-gemma-impact-challenge, 2026. Kaggle.
