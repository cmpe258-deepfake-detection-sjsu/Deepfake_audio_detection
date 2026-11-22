# AI Novelty & Feasibility Audit

This file has two parts. **Part 1** is the current audit, revised in November 2026 for the project as it now stands: shift decomposition on ASVspoof 5. **Part 2** is the original audit of the September proposal, kept unchanged so that the reasoning behind the pivot stays visible.

---

# Part 1 — Current audit (revised November 2026)

## 1. Why the audit was revised

The September proposal studied generalization to unseen attacks on ASVspoof 2019 LA, with attack-aware sampling, calibrated selective prediction and INT8 deployment as its three contributions. Prof. Liu's written feedback recommended modernising the evaluation around ASVspoof 5 and separating distribution shift into named mechanisms. The project was reframed around one question: **which shifts break modern detectors — unseen generators, codecs, or adversarial post-processing — and does an SSL front end fail differently from an older architecture?**

The Part 2 audit was written for a project that no longer exists, so its novelty and scope judgements had to be redone.

## 2. What changed from the original plan

| Original plan | Now | Reason |
|---|---|---|
| ASVspoof 2019 LA as the primary benchmark | Kept only to validate our evaluation code against the released AASIST checkpoint (0.83% EER) | ASVspoof 5 is the first public benchmark that labels generator, codec and adversarial shift on one evaluation set |
| Attack-aware sampling with leave-one-attack-out rotation on A01–A06 | Dropped | We use the released AASIST checkpoint rather than retraining it, so a training-side intervention no longer fits the question |
| ASVspoof 2021 DF for codec shift | Replaced by ASVspoof 5 codecs C01–C11 | Codec conditions in ASVspoof 5 can be paired with the same utterance uncoded, which 2021 DF does not allow |
| Calibrated selective prediction as the contribution | A diagnostic for RQ3 | Calibration and per-condition reporting are standard in the ASVspoof 5 results, and see §5 on temperature scaling |
| INT8 deployment as a required contribution | Optional; CPU latency and memory profiling only | It does not bear on the research question |
| DFADD as a secondary dataset | Kept, as an open-world check against recent diffusion and flow-matching TTS | Unchanged |

## 3. Current claim submitted for critique

> We decompose detector failure on ASVspoof 5 by cause and compare a specialised architecture (the released AASIST baseline) with a frozen SSL front end (XLS-R 300M with a lightweight head). Codec and adversarial shift are measured as matched contrasts with speaker-bootstrap intervals on the difference. Generator shift has no matched baseline on this benchmark and is reported as variation between families. The contribution is empirical and diagnostic; we propose no new architecture, training scheme or calibration method.

## 4. Red-ocean check for the new framing

Each of the following is established prior work, so none of it is claimed as ours.

- **Per-condition reporting on ASVspoof 5.** The organisers report per-condition results for submitted systems, and Dao et al. (Odyssey 2026) give a per-codec breakdown for SSL systems.
- **Decomposing the generalization gap.** Müller et al. (2024), *Harder or Different?*, split detector failure into hardness versus difference.
- **Shortcut learning on silence.** Müller et al. (Interspeech 2021) showed ASVspoof-trained models exploit leading and trailing silence. We treat this as a control to pass, not a finding.
- **Calibration in anti-spoofing.** Score calibration is standard in ASVspoof systems.
- **SSL front ends generalising better.** Widely reported. Our results agree, but the direction is not new.

What remains after removing those is narrower and, as far as our survey found, not done before:

- coded audio paired with **its own uncoded source utterance**, so that the codec effect can be split into what happens to real speech and what happens to fakes;
- adversarial shift placed in the **same decomposition**, as matched contrasts against the source attack;
- a **non-SSL reference point** scored on exactly the same slices as the SSL detector;
- error rates at a **fixed threshold selected on dev**, which show which class the errors fall in on each slice where EER cannot;
- **pre-registered controls** with falsification conditions fixed before the results.

## 5. Claims the critique corrected during the project

Most of the value of an audit like this came after the proposal, when results were in and the wording had to match them. These corrections came from re-reading our own outputs and from an independent AI review of our drafts. Each one weakened a claim we had already written.

1. **Ranking shift types by spread.** Our first headline ranked shift axes by the max-minus-min EER across conditions. That statistic depends on how many conditions sit in each group, so it cannot rank axes. It was replaced by two matched contrasts: A17 → A18 costs AASIST +43.28 EER points [41.84, 44.72] and XLS-R +2.22 [1.91, 2.58]; uncoded → C07 costs AASIST +7.41 [6.86, 7.93] and XLS-R +16.49 [15.86, 17.19].
2. **Calling generator differences "degradation".** The only seen-generator baseline (A01–A08) lives in the training partition, and the released AASIST checkpoint was trained on all of it. Generator results are now reported as variation between families.
3. **Expecting temperature scaling to improve abstention.** For binary logits, maximum-softmax confidence is monotonic in |z₁ − z₀|, so positive scalar temperature scaling leaves the confidence ordering unchanged and cannot move a risk–coverage curve at fixed coverage. The original Contribution 2 question ("does temperature scaling improve selective prediction?") therefore had a known answer of no. Calibration quality (ECE 0.0330 → 0.0268 on eval, T = 1.5223 fitted on dev) and ranking quality are now reported as separate questions.
4. **Ensembling three seeds.** On pooled eval the seeds span 7.272–7.663% EER and the ensemble lands at 7.435%, inside that range. All three seeds are reported alongside the ensemble, and no gain from ensembling is claimed.
5. **"Codecs degrade real speech, not spoofs".** Softened to "concentrated in the bona fide scores". Paired deltas are negative for real speech under all eleven codecs for AASIST and ten of eleven for XLS-R (C11 is +0.32). At a fixed threshold this holds for XLS-R, whose false rejection rises from 1.09% uncoded to 50.47% under C07 while false acceptance does not rise. It does not hold for AASIST under the neural codecs, where false acceptance rises by about 13 points.
6. **The silence control.** The first VAD-trimmed run showed two XLS-R codec effects changing. Diagnosis: the VAD ran separately on each condition and cut about 200 ms more from the 8 kHz files than from their uncoded sources. On the 2,718 utterances whose trim boundaries agree within 50 ms, every interval spans zero for both detectors. The matched subset is selected rather than random, so it is evidence about direction, not a revised effect size.
7. **Predicting where abstention helps.** We stated in advance that abstention should help on A18 and fail on A19 and A12. That was a hypothesis presented as an expectation; it was walked back and is being tested, with a null result reportable.

## 6. Threats to validity

- **Two detectors, one benchmark.** The two systems differ in front end, back end, training recipe and training subset, so differences are system-level, not isolated architectural causes. Conclusions are scoped to frozen-feature detectors; the systems leading ASVspoof 5 fine-tune the SSL front end.
- **Pretraining overlap.** XLS-R was pretrained on MLS, which shares LibriVox speech with ASVspoof 5. Part of its advantage may come from having seen the source corpus. LibriSpeech is also LibriVox-derived, so swapping backbones is not an overlap-free control.
- **Adversarial asymmetry.** The Malafide filters were optimised against an AASIST countermeasure, so for XLS-R they are transfer attacks. The adversarial contrast is reported as attack transfer, not as architectural robustness.
- **Mechanism.** XLS-R's two worst codecs (C04, C07) are the only neural codecs in the benchmark. That is a pattern, not a demonstrated cause. Dao et al. report the same ordering for a different SSL system.
- **DFADD.** Two test speakers and clean, uncoded audio. Intervals resample utterances and understate speaker variation.
- **Shortcuts beyond silence.** The VAD control rules out leading and trailing silence, not every non-speech cue. The silence-and-duration-only classifier has not been run.

## 7. Feasibility, as of November 20

The core results table, the matched contrasts, the paired codec analysis, the fixed-threshold error rates, DFADD for AASIST, and the random-score and boundary-matched VAD controls are complete and in the repository; the README status table lists each with its owner. Still open: per-slice calibration and risk–coverage in the repository, AASIST abstention analysis, XLS-R on DFADD, the silence-and-duration-only control, and CPU latency and memory profiling. None of these requires retraining.

## 8. Revised assessment

**Novelty:** Modest, at the level of evaluation design. The contribution is the paired, matched decomposition and the cross-architecture comparison on it, not a method.

**Red-ocean risk:** Low for the current claim, because every established component above is credited rather than claimed. It would be high if the project were described as per-codec reporting or as a calibration study.

**Biggest danger now:** Overgeneralising from two detectors on one benchmark. "SSL front ends fail on neural codecs" is not supported; "this frozen XLS-R detector is most hurt by the two neural codecs in ASVspoof 5, through false rejection of real speech" is.

**Recommended positioning:**

> We do not propose a better detector. We show that a single pooled EER hides opposite failure modes: on matched contrasts, adversarial post-processing breaks the AASIST baseline while neural codecs break a frozen XLS-R detector, and at a fixed threshold the XLS-R codec failure is real speech being rejected, not fakes being accepted.

## 9. Recommendation

**Continue, with scope frozen.** No new experiments beyond the open items in §7. Remaining effort goes to the write-up, with negative results and threats to validity reported as prominently as the positive findings.

---

# Part 2 — Original audit of the September proposal (unchanged)

## 1. Project Claim Submitted for Critique

The initial proposal framed the project as generalizable audio deepfake detection under unseen attacks, with attack-aware training, calibration/selective prediction, and INT8 deployment.

The intended novelty was originally described as "frozen-calibration selective prediction under unseen attacks."

## 2. AI Critique Summary

### Red-ocean areas

**A. Training another AASIST-style detector**

This is highly saturated. ASVspoof is a mature challenge ecosystem with many closely related detector architectures and system submissions.

**B. Generalization to unseen attacks as the sole contribution**

This is not sufficiently novel by itself. Recent datasets and challenge protocols explicitly target robustness and generalization to new synthesis methods.

**C. Spectro-temporal fusion as an improvement**

Not appropriate as a novelty claim because AASIST already provides an established spectro-temporal graph-attention approach.

**D. Codec augmentation as the primary innovation**

Also weak as a novelty claim because recent ASVspoof systems already use codec and other augmentation strategies.

**E. Competing primarily through dataset scale**

Large contemporary datasets and broad training mixtures are a different resource regime from this course project and are not a practical direction for a three-person team with consumer-GPU constraints.

## 3. Important Revision to the Original Novelty Claim

The literature review revealed that **calibration itself is not new in speech anti-spoofing**. For example, recent ASVspoof work has used score calibration and fusion.

Likewise, recent work such as PV-VASM explicitly studies probabilistic verification under unseen synthesis conditions.

Therefore, we should **not** claim:

> "No prior speech anti-spoofing work uses calibration."

We should instead claim the narrower and more defensible question:

> **Does post-hoc temperature scaling improve selective prediction under unseen attack shift, when evaluated using risk-coverage and AURC, compared with maximum-softmax probability?**

This is an empirical research question, not a claim that the underlying calibration algorithm is novel.

## 4. Revised Contribution

The project has three experimentally separable components:

### Contribution 1 — Attack-aware sampling

Compare random sampling with attack-balanced batches across known attack types.

Controlled test:
- train on five of A01–A06
- evaluate on the sixth
- rotate across all six attacks

This isolates whether attack diversity in training improves transfer.

### Contribution 2 — Calibrated selective prediction

Fit temperature scaling only on the official development set.

Then:
- freeze calibration;
- evaluate unseen attacks;
- compare calibrated confidence with MSP;
- construct risk-coverage curves;
- report AURC and ECE.

The key question is whether confidence quality remains useful when the attack distribution changes.

### Contribution 3 — Efficient deployment

Quantize the selected model to INT8 and measure:
- CPU latency
- model size
- EER
- absolute ΔEER

## 5. Feasibility Audit

### Compute

**Feasible.**

The plan uses two primary model families rather than a large architecture search. The project targets a single consumer GPU for training and CPU inference after quantization.

### Data

**Feasible.**

The project is restricted to three public datasets:

- ASVspoof 2019 LA
- ASVspoof 2021 DF
- DFADD

The official ASVspoof protocol remains the primary benchmark.

### Experimental complexity

**Moderate but manageable.**

The critical dependency is the preprocessing/data pipeline. Once the pipeline is fixed, the three members can work in parallel on:

- baseline/training
- calibration/selective prediction
- cross-dataset/deployment evaluation

### Timeline

**Main risk: baseline reproduction.**

The project should have a hard checkpoint after the first three weeks:

> If AASIST cannot be reproduced under the intended protocol, stop adding proposed components and debug the baseline first.

This prevents unreliable baseline comparisons from contaminating the rest of the project.

## 6. Scope Controls

To prevent scope creep:

### Required
- AASIST baseline
- SSL baseline
- attack-aware sampling
- temperature scaling
- MSP comparison
- risk-coverage/AURC
- ECE/reliability diagrams
- ASVspoof 2019 LA
- INT8 latency/size/ΔEER

### Secondary
- ASVspoof 2021 DF
- DFADD

### Optional only if the required experiments are complete
- energy-based OOD scoring

### Explicitly out of scope
- training a new TTS/VC generator
- large-scale synthetic-data generation
- architecture search
- large foundation-model fine-tuning
- competing through massive training-data scale
- adding multiple OOD/uncertainty algorithms without a clear experimental need

## 7. Final AI Assessment

**Novelty:** Moderate, primarily at the evaluation/experimental-design level rather than algorithmic novelty.

**Red-ocean risk:** High if framed as "another audio deepfake detector"; substantially lower if framed around calibrated selective prediction under attack shift.

**Technical depth:** Strong enough for a deep-learning course because it combines representation learning, controlled training changes, calibration, selective prediction, robustness evaluation, and deployment.

**Feasibility:** Good for a three-person team if the baseline and data pipeline are completed first.

**Biggest danger:** Overclaiming novelty.

**Recommended positioning:**

> We are not proposing a fundamentally new anti-spoofing architecture. We are conducting a controlled study of whether attack-aware training and post-hoc calibrated selective prediction make a practical anti-spoofing detector more reliable when its attack distribution changes, while also measuring deployment cost.

## 8. Go/No-Go Recommendation

**GO**, with the revised novelty framing.

The project is sufficiently differentiated for a course project if the proposal emphasizes:
1. unseen-attack evaluation,
2. confidence quality rather than EER alone,
3. abstention/risk-coverage behavior,
4. controlled ablations, and
5. deployment constraints.

The project should not promise that calibration or attack-aware sampling is a new algorithm. The research value comes from the controlled question and the evidence produced by the experiments.
