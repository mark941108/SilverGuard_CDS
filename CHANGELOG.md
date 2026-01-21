# Changelog

All notable changes to the **AI Pharmacist Guardian** project will be documented in this file.

## [V7.1] - Impact Edition - 2026-01-20
### Fixed
-   **Aspirin Safety Logic:** Refined logic to distinguish between standard preventive dosage (100mg) and high-risk analgesic dosage (>325mg) for elderly patients, aligning with **AGS Beers Criteria 2023**.
-   **Warfarin Lookup Bug:** Fixed a reverse lookup issue where the generic name was not correctly mapping to the brand name in the safety check.

### Changed
-   **UI Optimizations:** Improved SilverGuard font rendering for better accessibility.

## [V6.0] - Agentic Pipeline - 2026-01-15
### Added
-   **Self-Correction Loop:** Implemented the agentic retry mechanism.
-   **Dynamic Temperature:** System now automatically lowers temperature from `0.6` to `0.2` when a logical flaw is detected during the first inference pass.

### Security
-   **Input Gate:** Added Laplacian Variance check to reject blurry or non-prescription images before processing.

## [V5.0] - Data Engine - 2026-01-10
### Added
-   **"Gallery of Horrors":** Introduced a stress-test dataset containing extreme edge cases (blur, heavy noise, occlusion, rotation) to improve Sim2Real robustness.
-   **Risk Injection:** Training data now includes 30% "dangerous" examples (e.g., elderly overdose scenarios) to train the model's safety discrimination capability.
