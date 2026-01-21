### Project name
**SilverGuard: AI Pharmacist Guardian (V5.0 Impact Edition)**

### Track Selection
* **Main Track**
* **Special Award:** Agentic Workflow Prize

### Links
* **Video Demo:** [INSERT YOUTUBE/VIDEO LINK HERE] (Must be < 3 mins)
* **Code Repository:** [INSERT GITHUB/KAGGLE NOTEBOOK LINK HERE]
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
