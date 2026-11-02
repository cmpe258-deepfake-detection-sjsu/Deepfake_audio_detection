# Literature & SOTA Survey

## Purpose

This survey positions the project against recent work in speech deepfake detection, self-supervised representations, generalization under distribution shift, adversarial attacks on countermeasures, and shortcut learning. AASIST is included as the foundational baseline even though it predates the required recent-paper window.

*Entries 1-9 were written for the original proposal and are unchanged. Entries 10-14 were added after the project was refocused on shift decomposition over ASVspoof 5, and cover the areas that refocus made load-bearing: generalization decomposition, shortcut learning, the adversarial attacks in the ASVspoof 5 evaluation set, and codec robustness.*

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

### 10. Müller, Czempin, Dieckmann, Froghyar & Böttinger, "Harder or Different? Understanding Generalization of Audio Deepfake Detection," 2024 (arXiv:2406.03512)

Splits the in-domain/out-of-domain performance gap into a **hardness** component (the new data is intrinsically harder) and a **difference** component (the model fails to transfer), across four detectors trained on ASVspoof 2019 LA and tested on ASVspoof 2021 LA/DF, truncated audio and In-the-Wild. Finds that difference dominates for unseen attacks and for telephony/compression, and that SSL-based models show smaller gaps, which the authors tentatively attribute to less training-specific features.

**Relevance:** this is the closest prior work to our framing. It decomposes generalization failure rather than reporting a pooled number, and it compares SSL against specialised architectures.

**Difference from our approach:** the split is hardness versus difference, not *by cause*. The paper does not separate unseen generator from codec from adversarial post-processing, does not use ASVspoof 5 (where those three are labelled on one evaluation set), and does not pair coded audio with its own uncoded original. Its SSL observation is explicitly described as preliminary.

**Implication:** we must not claim that decomposing the generalization gap is new. Our contribution is the per-cause decomposition on a benchmark that labels all three shifts, with utterance-level pairing and speaker-level intervals, and the mechanism we identify for *why* one architecture fails.

---

### 11. Müller, Dieckmann, Czempin, Canals, Böttinger & Williams, "Speech is Silver, Silence is Golden: What do ASVspoof-trained Models Really Learn?," Interspeech 2021 (arXiv:2106.12914)

Shows that models trained on ASVspoof rely heavily on the **leading and trailing silence** of an utterance rather than on the speech itself, to the point that detection performance can largely be reproduced from non-speech regions alone.

**Relevance:** this is the canonical shortcut result in our field, and it is the reason our controls table includes a VAD-trimmed rerun and a silence-and-duration-only classifier. It also bears directly on our finding that codec conditions degrade the bona fide class rather than hiding spoof artifacts: if a detector keys on signal quality rather than synthesis evidence, both results follow.

**Implication:** any claim we make about a codec effect has to survive VAD trimming before we can attribute it to speech content. This is stated as a falsification condition in the project README, not added after the fact.

---

### 12. Panariello, Todisco & Evans, "Malafide: a novel adversarial convolutive noise attack against deepfake and spoofing detection systems," Interspeech 2023 (arXiv:2306.07655)

Introduces a convolutive noise filter optimised to defeat a specific spoofing countermeasure while leaving the attacked utterance's speaker characteristics intact.

**Relevance:** Malafide is applied in the ASVspoof 5 evaluation set as attacks A18, A20, A23 and A30, which form part of our adversarial slice. The filters released with the benchmark were optimised **against an AASIST countermeasure**.

**Implication:** our adversarial comparison is not symmetric. For AASIST these are near-white-box attacks; for the SSL detector they are transfer attacks. Any cross-architecture statement about adversarial robustness has to be reported as attack transfer, not as an architecture property.

---

### 13. Todisco, Panariello, Wang, Delgado, Lee & Evans, "Malacopula: adversarial automatic speaker verification attacks using a neural-based generalised Hammerstein model," ASVspoof Workshop 2024 (arXiv:2408.09300)

A non-linear filter optimised to make spoofed speech better match a target speaker, aimed at the **ASV system** rather than at the countermeasure.

**Relevance:** Malacopula appears in ASVspoof 5 as A27, A31 and A32, and as the second stage of A30. Because its target is ASV rather than the countermeasure, it sits in our adversarial slice with a different threat model from Malafide, and the two should not be pooled.

**Implication:** A30 is the only ASVspoof 5 attack applying both filters, which makes it the natural worst case and worth reporting separately rather than averaging into a single adversarial number.

---

### 14. Dao, Matrouf, Rouvier & Evans, "A Comparison of SSL-Based Feature Extractors and Back-End Classifiers for Spoofing Detection: A Multi-Corpus Training and Cross-Linguistic Analysis," Odyssey 2026, pp. 263-268 (DOI: 10.21437/Odyssey.2026-39)

Compares SSL front ends and back-end classifiers for spoofing detection across corpora. Table 4 reports ASVspoof 5 EER separately for uncoded audio and each codec condition C01-C11, for an XLSR-ResNet system under two training settings:

| Condition | ASVspoof 5 training only | ASVspoof 5 + MLAAD-v3 |
|---|---|---|
| uncoded | 0.95% | 0.53% |
| C04 (Encodec) | 11.87% | 27.11% |
| C07 (mp3+Encodec) | 14.52% | 32.32% |
| pooled | 4.72% | 12.34% |

**Relevance:** this is the nearest prior work on our codec section, and it agrees with our codec ordering. A different SSL front end, a different back end and different training data produce the same ranking: C04 and C07 are the two worst codec conditions, and C07 is worse than C04. In our run the same two conditions are XLS-R's worst, at +16.49 and +13.21 EER points against the same source utterances uncoded. This is agreement on the ordering, not a replication of our paired class-specific analysis, which they do not perform.

Their second column is consistent with the mechanism we propose, though the reading is ours rather than theirs. Adding MLAAD training data *improves* uncoded performance while roughly doubling the error under both neural codecs. If the failure were simply that compression destroys information, more training data should not make it worse. If instead the detector is learning a sharper model of what genuine speech looks like, then audio re-synthesised through a neural codec violates that model harder the sharper it gets - which is the direction their numbers move.

**Difference from our approach:** Dao et al. report pooled per-condition EER for SSL systems only. That cannot distinguish a codec that makes spoofs look real from one that makes real speech look fake, and it offers no non-SSL reference point against which to ask whether the vulnerability is specific to SSL representations.

**Implication:** per-codec reporting on ASVspoof 5 is established and must not be claimed as our contribution. What we add is the paired decomposition - each coded file against its own uncoded original, split by class - and the comparison against a specialised architecture whose codec failures fall on entirely different conditions (the 8 kHz DSP codecs, not the neural ones).

---

## SOTA Positioning

| Area | Recent direction | Our treatment |
|---|---|---|
| Specialized detector architecture | AASIST, AASIST2, AASIST3 | Released AASIST checkpoint used as the non-SSL reference point |
| SSL representations & efficient fine-tuning | WavLM, XLS-R, MoLEx (LoRA+MoE) | Frozen XLS-R with a lightweight head; **not** parameter-efficient fine-tuning, and our conclusions are scoped to frozen-feature detectors |
| Generalization decomposition | Hardness-vs-difference split (Harder or Different?) | Decompose **by cause** instead: generator family, codec condition, adversarial post-processing, each against a matched reference |
| Unseen-attack generalization | DFADD, ASVspoof 5, cross-domain studies | One of three shift types, not the whole setting |
| Codec / compression robustness | Per-condition breakdowns on ASVspoof 5 | Utterance-level pairing against the uncoded original, decomposed per class |
| Adversarial attacks on countermeasures | Malafide (anti-CM), Malacopula (anti-ASV) | Evaluated as a separate shift type; reported as attack transfer for the SSL detector, since the released filters target AASIST |
| Shortcut learning | Silence and duration cues (Speech is Silver...) | Pre-registered VAD-trimmed and silence-only controls as falsification conditions |
| Data augmentation | Front-/back-end + augmentation studies | Out of scope; no augmentation in the base condition |
| Layer/representation selection | Probing-guided layer selection | Not our contribution; noted as an alternative SSL-side lever |
| Robustness/uncertainty | PV-VASM provides formal probabilistic verification | Calibration and risk-coverage used as a **diagnostic** per shift type, not as the contribution |
| Efficient deployment | Parameter-efficient/SSL-based systems are an active area | INT8 + CPU latency, optional and only after the core table exists |

## Research Gap

The literature shows that:

1. unseen-attack and cross-domain generalization are established challenges, not a novel framing;
2. SSL representations are the dominant direction for strong detectors, and are already reported to generalize better, though the reason is not established;
3. architecture variants (AASIST2/3) and data augmentation are heavily explored;
4. decomposing the generalization gap has been attempted, but by *hardness versus difference* rather than by cause, on benchmarks that do not label generator, codec and adversarial shift on one evaluation set;
5. shortcut learning on non-speech regions is a known failure mode of ASVspoof-trained detectors; and
6. calibration and per-condition reporting are already standard in the ASVspoof 5 results.

Therefore our novelty claim is deliberately narrow:

> **We decompose detector failure on ASVspoof 5 by cause, and compare a specialised architecture against a frozen SSL front end. Codec and adversarial shift admit matched contrasts — the same utterance coded and uncoded, the same generator with and without an adversarial filter — and are reported as differences with speaker bootstraps on the difference. Generator shift admits no valid baseline on this benchmark and is reported as between-family variation. We ask whether the two architectures degrade in the same order under the matched contrasts, and if not, what mechanism explains the difference.**

This is an **empirical/diagnostic contribution**. We do not propose a new architecture, training scheme or calibration algorithm.

What the decomposition adds beyond existing per-condition reporting:

- a non-SSL reference point evaluated on the same slices as the SSL detector, which existing per-codec reporting (Dao et al.) does not provide;
- **matched contrasts with intervals on the difference**: each codec against the same source utterances uncoded, and each Malafide attack against its own source attack, rather than comparing two separately-estimated EERs;
- **utterance-level pairing** of coded audio with its own uncoded original, which allows the codec effect to be split by class rather than summarised as one EER;
- adversarial conditions placed inside the same decomposition rather than reported separately;
- calibration and risk-coverage computed **per shift type**, to ask whether confidence fails together with accuracy or independently of it;
- pre-registered controls, including the VAD-trimmed rerun, with falsification conditions fixed in advance.

A negative result remains useful: if both architectures degrade in the same order with overlapping intervals, pooled reporting is adequate and we will say so.
