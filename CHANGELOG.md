# Changelog

All notable changes to the **AI Pharmacist Guardian** project will be documented in this file.

## [V8.2] - Deployment Hardening - 2026-01-29
### Critical Infrastructure Fixes
-   **Cloud Deployment Stablity**: Added `espeak`, `libespeak1`, and `ffmpeg` to `packages.txt` and `Dockerfile`. This resolves the `OSError: libespeak.so.1` crash when `pyttsx3` attempts offline TTS in Linux/Docker environments (Hugging Face Spaces).
-   **Dependency Robustness**: Explicitly added `tesseract-ocr` and `libtesseract-dev` to ensure OCR capabilities are self-contained in the container.

## [V8.1] - Code Audit Hotfix - 2026-01-28
### Security & Logic Fixes
-   **Critical**: Fixed `get_confidence_status` label mismatch. Added support for V8 `WITHIN_STANDARD` label (threshold 0.75) and `PHARMACIST_REVIEW_REQUIRED` (threshold 0.50).
-   **Hard Rule Robustness**: Replaced fragile string matching (`if "2000" in dose`) with robust Regex mathematical verification to catch `1000mg BID` sum violations.
-   **Anti-Hallucination**: Implemented "Post-Hoc RAG Verification". If Agent misses RAG context on first pass but detects a drug name, forced retry with injected knowledge is triggered.

### UX Improvements
-   **Async UI Non-Blocking**: Wrapped `pyttsx3` offline TTS in `asyncio.to_thread` to prevent Gradio UI freeze during voice generation.
-   **Consistency**: Synced `HF_SPACE_APP.py` and `generate_stress_test.py` with `medgemma_data.py` Source of Truth.

## [V7.2] - Competition Polish - 2026-01-23
### Fixed
-   **Critical Syntax Error:** Removed dangling docstring at line 379 causing Python SyntaxError.
-   **f-string Escape Bug:** Fixed broken escape characters in Temperature Shift logging (line 1471).
-   **mcg/ug Unit Conversion:** Proper handling of micrograms (1000mcg = 1mg) to prevent false positives on B12 vitamins.
-   **Decimal Dose Regex:** Updated regex to support `0.5mg` and other decimal dosages (`[\d.]+` instead of `\d+`).

### Changed
-   **README Architecture Diagram:** Replaced with Neuro-Symbolic Mermaid flowchart showing explicit retry loop path.
-   **WRITEUP LaTeX Formulas:** Added formal mathematical definitions for Temperature Policy and Confidence Threshold.
-   **Google Ecosystem Enhancements:** Added SigLIP praise and TFLite/MediaPipe roadmap to strengthen Google alignment.

### Added
-   **Anti-Fragility Section:** New "Limitations & Mitigations" documentation with defensive narrative strategy.
-   **Consumer GPU Claims:** Documented RTX 3060/4060 support with ~2-3 sec inference latency.

## [V5.0] - Impact Edition (Final Submission) - 2026-01-20
### Fixed
-   **Aspirin Safety Logic:** Refined logic to distinguish between standard preventive dosage (100mg) and high-risk analgesic dosage (>325mg) for elderly patients, aligning with **AGS Beers Criteria 2023**.
-   **Warfarin Lookup Bug:** Fixed a reverse lookup issue where the generic name was not correctly mapping to the brand name in the safety check.

### Changed
-   **UI Optimizations:** Improved SilverGuard font rendering for better accessibility.

## [V4.5] - Agentic Pipeline - 2026-01-15
### Added
-   **Self-Correction Loop:** Implemented the agentic retry mechanism.
-   **Dynamic Temperature:** System now automatically lowers temperature from `0.6` to `0.2` when a logical flaw is detected during the first inference pass.

### Security
-   **Input Gate:** Added Laplacian Variance check to reject blurry or non-prescription images before processing.

## [V5.0] - Data Engine - 2026-01-10
### Added
-   **"Gallery of Horrors":** Introduced a stress-test dataset containing extreme edge cases (blur, heavy noise, occlusion, rotation) to improve Sim2Real robustness.
-   **Risk Injection:** Training data now includes 30% "dangerous" examples (e.g., elderly overdose scenarios) to train the model's safety discrimination capability.
