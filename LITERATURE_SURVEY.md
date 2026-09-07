# Literature & SOTA Survey

## Purpose

This survey positions the project against recent work in speech deepfake detection, self-supervised representations, generalization to unseen attacks, and confidence/robustness evaluation. AASIST is included as the foundational baseline even though it predates the required recent-paper window.

## Foundational Baseline

### Jung et al., "AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks," ICASSP 2022

AASIST is the primary specialized anti-spoofing baseline. It uses graph attention to model spectro-temporal relationships in speech. We reproduce it as a reference point rather than claiming spectro-temporal fusion as our contribution.

**Why it matters:** establishes the architecture baseline against which the proposed training and decision-level changes are evaluated.

---

## Recent Literature

### 1. Du et al., "DFADD: The Diffusion and Flow-Matching Based Audio Deepfake Dataset," IEEE SLT 2024

DFADD was designed to expose weaknesses of existing anti-spoofing systems against newer diffusion- and flow-matching-based TTS systems. The dataset contains spoofed speech from five modern TTS systems and is intended to probe robustness beyond traditional spoofing benchmarks.

**Relevance:** directly motivates our secondary open-world evaluation.

**Limitation for our project:** DFADD is primarily a dataset/robustness resource, not a solution to uncertainty-aware decision making.

---

### 2. Zhang et al., "Improving Short Utterance Anti-Spoofing with AASIST2," ICASSP 2024

AASIST2 modifies AASIST with multi-scale feature extraction and training strategies intended to improve performance on short utterances.

**Relevance:** demonstrates that architectural and training changes can improve anti-spoofing performance without changing the overall task.

**Implication for our project:** our work should distinguish training-strategy effects from architecture changes and avoid presenting another AASIST architectural variant as the main contribution.

---

### 3. Borodin et al., "AASIST3: KAN-Enhanced AASIST Speech Deepfake Detection Using SSL Features and Additional Regularization," ASVspoof 2024

AASIST3 combines Kolmogorov-Arnold Network (KAN) layers with SSL front-end features and extra regularization, submitted to the ASVspoof 2024/5 Challenge.

**Relevance:** a second, more recent architectural-variant data point beyond AASIST2, useful for calibrating how much architecture alone moves the needle versus our training/decision-level changes.

**Implication:** reinforces that architecture search on top of AASIST is an active, somewhat saturated sub-area — another reason to keep our contribution at the training-strategy and calibration level.

---

### 4. Schäfer, Choi & Neu, "Robust Audio Deepfake Detection: Exploring Front-/Back-End Combinations and Data Augmentation Strategies for the ASVspoof5 Challenge," ASVspoof 2024

Evaluates AASIST(-L)/RawGAT-ST versus SSL-based front ends under different augmentation strategies, finding that heavy augmentation helps non-SSL front ends but can hurt SSL-based ones.

**Relevance:** a recent, robustness-oriented comparison that helps prevent us from treating augmentation as a novel contribution, and shows front-end choice interacts with augmentation in non-obvious ways.

**Implication:** supports keeping augmentation out of scope and focusing on attack-aware sampling and calibration instead.

---

### 5. Wang et al., "ASVspoof 5: Crowdsourced Speech Data, Deepfakes, and Adversarial Attacks at Scale," 2024

ASVspoof 5 expands the benchmark to diverse crowdsourced speech, many speakers, dozens of attack algorithms, and adversarial attacks, explicitly emphasizing robust speech deepfake detection.

**Relevance:** establishes the current benchmark direction and confirms the field is actively moving toward more diverse and difficult attack conditions.

**Implication:** generalization to unseen attacks is an established research problem, not our novelty claim — our proposal already treats it as the baseline expectation, not the contribution.

---

### 6. Pan, Bhupendra & Wu, "MoLEx: Mixture of LoRA Experts in Speech Self-Supervised Models for Audio Deepfake Detection," 2025

Combines Low-Rank Adaptation with a Mixture-of-Experts router on top of a frozen SSL model, reducing fine-tuning cost while letting the router activate different experts for different attack types; reports SOTA EER on ASVspoof 5.

**Relevance:** a recent, more sophisticated alternative to a plain "SSL + lightweight head" baseline, and evidence that attack-type-aware behavior (their router specialization) is an active research direction — conceptually adjacent to our attack-aware sampling, but at the architecture level rather than the training-strategy level.

**Implication:** we should be explicit that our SSL baseline is intentionally simple (frozen front end, lightweight head) rather than a parameter-efficient fine-tuning scheme like MoLEx, since matching MoLEx's approach is out of scope for a course project.

---

### 7. Ahmadiadli, Zhang & Khan, "Beyond Identity: A Generalizable Approach for Deepfake Audio Detection," 2025

Targets cross-generator generalization explicitly (as opposed to speaker-identity artifacts), reinforcing that generalization-to-unseen-attacks remains the field's acknowledged central open problem rather than a novel framing on its own.

**Relevance:** shows that recent work is actively addressing generalization by changing what representations the detector learns.

**Implication:** our project should not claim that cross-generator generalization itself is new. Instead, we study whether a lightweight training strategy plus calibrated selective prediction can provide useful uncertainty information under that shift.

---

### 8. Beheshti, Rostami & Chen, "Probing-Guided Layer Selection from Self-Supervised Speech Models for Generalizable Audio Deepfake Detection," 2026

Studies which layers of SSL speech models transfer across domains and proposes probing-guided layer selection for generalizable detection, using lightweight XGBoost probes to rank transformer layers before any task-specific model is trained.

**Relevance:** demonstrates that representation choice is an active research direction for cross-domain generalization.

**Implication:** it strengthens the case for including an SSL baseline while keeping our main experimental axis at the training/decision level rather than proposing another SSL architecture or layer-selection scheme.

---

### 9. Kushnir et al., "Probabilistic Verification of Voice Anti-Spoofing Models," 2026

PV-VASM proposes a probabilistic framework (built on concentration inequalities) for estimating misclassification probability under unseen transformations and speech-generation techniques, including a theoretical upper bound, validated against several TTS/VC generators.

**Relevance:** this is the closest adjacent work, since it explicitly addresses uncertainty/robustness for unseen synthesis conditions.

**Difference from our approach:** PV-VASM is a probabilistic verification framework with formal robustness bounds, evaluated per-input against specific transformations/generators. Our project uses standard post-hoc temperature scaling and selective prediction, then empirically evaluates risk versus coverage across attack families — a simpler, evaluation-driven approach rather than a certification framework.

**Implication:** this paper prevents us from overstating novelty. Our contribution should be framed as a controlled empirical study of calibration/selective prediction under unseen attacks, not as the first work to consider uncertainty or robustness verification.

---

## SOTA Positioning

| Area | Recent direction | Our treatment |
|---|---|---|
| Specialized detector architecture | AASIST, AASIST2, AASIST3 | Reproduce AASIST as baseline |
| SSL representations & efficient fine-tuning | WavLM, XLS-R, MoLEx (LoRA+MoE) | Include a simple SSL + lightweight-head baseline (not parameter-efficient fine-tuning) |
| Unseen-attack generalization | DFADD, ASVspoof 5, cross-domain studies (Beyond Identity) | Primary research setting |
| Data augmentation | Front-/back-end + augmentation studies (Schäfer et al.) | Not claimed as novelty |
| Layer/representation selection | Probing-guided layer selection | Not our contribution; noted as an alternative SSL-side lever |
| Robustness/uncertainty | PV-VASM provides formal probabilistic verification | Use a simpler empirical calibration + risk-coverage framework |
| Efficient deployment | Parameter-efficient/SSL-based systems are an active area | INT8 + CPU latency as our deployment constraint |

## Research Gap

The literature shows that:

1. unseen-attack and cross-domain generalization are established challenges;
2. SSL representations, and increasingly parameter-efficient fine-tuning on top of them, are a major direction for strong detectors;
3. architecture variants (AASIST2/3) and data augmentation are already heavily explored;
4. representation/layer selection is an emerging direction for generalization; and
5. probabilistic robustness verification for unseen conditions is emerging, but via formal bounds rather than calibrated selective prediction.

Therefore, our novelty claim is deliberately narrow:

> **We evaluate whether post-hoc calibrated confidence can support useful selective prediction under unseen attack shift, and whether attack-aware sampling improves that behavior, while holding the official benchmark protocol fixed.**

This is an **empirical/evaluation contribution**, not a claim of a fundamentally new calibration algorithm.

The key comparison is:

**uncalibrated MSP → temperature-scaled confidence → risk/coverage**

under:

**in-domain → unseen attack → codec shift → cross-dataset shift**.

A positive or negative result is useful because the project explicitly tests whether calibration and abstention provide value beyond ordinary EER reporting.
