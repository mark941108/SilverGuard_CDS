---
description: Kaggle 比賽提交前的最終審查與優化流程 (Pre-submission optimization workflow for Kaggle)
---

# 🏆 Kaggle Competition Pre-Submission Optimization Skill

This skill provides a systematic methodology for auditing and optimizing a Kaggle competition project before final submission. It was developed during the MedGemma Impact Challenge (2026) and is applicable to any AI/ML competition.

## When to Use This Skill

- Before submitting to any Kaggle competition with complex submission requirements (code, writeup, video)
- When you need to maximize your chances for "Special Prizes" (e.g., Agentic Workflow Prize, Social Impact Prize)
- When you want a rigorous "red team" audit of your project

---

## Phase 1: Persona-Based Attack Simulation

Create two extreme personas in NotebookLM (or manually simulate) to stress-test your project from opposing angles:

### 1.1 Dr. K (The Cynical Technical Judge)

```
You are "Dr. K", a cynical Kaggle Grandmaster and Senior AI Researcher. You have judged 50+ competitions and despise "hype" and "buzzwords".

Your Goal: Tear apart this project's technical claims.

Review the uploaded documents and provide a brutal critique focusing on:
1.  **Architecture**: Is the core claim technically sound, or is it just a fancy wrapper?
2.  **Safety/Robustness**: Are there hardcoded values (magic numbers) or fragile regex patterns?
3.  **Data**: Is the evaluation methodology valid, or is it "Sim2Sim" (training and testing on the same distribution)?
4.  **Code Quality**: Are there any "smells" that suggest amateur implementation?

End with a verdict: [ACCEPT] or [REJECT] and explain why.
```

### 1.2 Sarah (The Pragmatic VC Investor)

```
You are "Sarah", a pragmatic Healthcare/Tech Venture Capitalist. You don't care about code loops; you care about User Outcomes, Scalability, and ROI.

Your Goal: Evaluate if this project solves a real-world problem.

Review the uploaded documents and provide an investment memo focusing on:
1.  **Problem-Solution Fit**: Does this solve a burning pain point better than existing solutions?
2.  **User Experience**: Is the UX genuinely designed for the target user, or is it an afterthought?
3.  **Cost Structure**: Is the proposed deployment financially sound?
4.  **Risks**: What are the regulatory or real-world deployment risks?

End with a verdict: [INVEST] or [PASS] and summarize the "Secret Sauce".
```

### Action Items from Personas

- **For Dr. K's [REJECT]**: Identify specific code lines to fix (e.g., expand regex, add guardrails, extract magic numbers as constants with citations).
- **For Sarah's [PASS]**: Identify missing "Impact" narrative. Add hooks like "Privacy Moat," "Unit Economics," or "Last Mile Problem."

---

## Phase 2: Documentation Consistency Audit

Run the following consistency checks across all documentation files (`README.md`, `WRITEUP.md`, code files):

### 2.1 Numerical Consistency

Search for key statistics and ensure they match the actual code:

```bash
# Example: Search for drug count claims
grep -rn "Drug Types" README.md WRITEUP.md
# Verify against code
grep -rn "DRUG_DATABASE" *.py | wc -l
```

### 2.2 Version Consistency

Ensure library versions are consistent across all `requirements.txt` files:

```bash
grep -rn "transformers" *.txt *.py
```

### 2.3 Placeholder Link Check

Find all placeholder links that need to be filled before submission:

```bash
grep -rn "Insert\|TODO\|PLACEHOLDER" *.md
```

---

## Phase 3: Code Robustness Improvements (Dr. K Fixes)

Apply the following "Dr. K Fix" patterns identified during the MedGemma challenge:

| Pattern | Description | Example |
| :--- | :--- | :--- |
| **Regex Expansion** | Expand regex patterns to cover more real-world cases. | `mg` → `mg\|ml\|tablet\|capsule\|pill\|drops` |
| **Edge Case Guardrails** | Add explicit checks for unexpected input ranges. | `if age < 18 or age > 120: trigger_human_review()` |
| **Magic Number Extraction** | Extract hardcoded numbers as named constants with documentation. | `BLUR_THRESHOLD = 100 # Reference: pyimagesearch.com` |
| **Alias Mapping** | Create lookup tables for synonyms (e.g., brand/generic drug names). | `DRUG_ALIASES = {"glucophage": "metformin", ...}` |
| **OOD Detection** | Strengthen Out-of-Distribution input detection. | Increase keyword threshold from 2 to 3 for stricter validation. |

---

## Phase 4: Strategic Narrative Enhancement (Sarah Hooks)

Insert the following "VC Power Hooks" into your Video Script and Writeup:

| Hook | Description | Example Usage |
| :--- | :--- | :--- |
| **Privacy Moat** | Emphasize data isolation advantages over cloud APIs. | "Unlike GPT-4V, our Edge AI never sends PHI to the cloud." |
| **Digital Safety Net** | Reframe "loop" as "fail-safe architecture." | "It's not just a loop; it's a safety net that catches errors before they reach patients." |
| **Unit Economics** | Quantify cost savings. | "Cloud API = $300/month. Edge = $0/month after hardware." |
| **Secret Sauce** | Summarize the unique value proposition. | "The Secret Sauce? It's the Neuro-Symbolic Fail-Safe." |

---

## Phase 5: Final Pre-Submission Checklist

- [ ] **Code Execution**: Run all cells in Kaggle and verify outputs (especially demo cells).
- [ ] **Secrets**: Ensure `HUGGINGFACE_TOKEN` (or other secrets) are set in Kaggle Secrets.
- [ ] **Public Setting**: Set Notebook and Datasets to "Public".
- [ ] **Placeholder Links**: Fill in all `[Insert Link]` placeholders in Writeup.
- [ ] **Model Upload**: Verify model is uploaded to Hugging Face Hub (Bonus 1).
- [ ] **Space Deployment**: Verify Hugging Face Space is live (Bonus 2).
- [ ] **Video**: Record demo video following the script (use ElevenLabs for AI narration).
- [ ] **Submission Form**: Select correct track (e.g., "Agentic Workflow Prize").

---

## Origin

This skill was developed during the **MedGemma Impact Challenge (January 2026)**. It synthesizes lessons learned from:

1.  Simulating a "Dr. K" Kaggle Grandmaster attack to expose technical weaknesses.
2.  Simulating a "Sarah" Healthcare VC review to identify impact gaps.
3.  Systematically patching 5 code vulnerabilities and aligning 3 documentation narratives.
