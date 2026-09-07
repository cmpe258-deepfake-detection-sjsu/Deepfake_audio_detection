# AI Novelty & Feasibility Audit

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
