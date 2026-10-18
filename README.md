# What Breaks Modern Audio Deepfake Detectors?

**Generator, codec, and adversarial shift on ASVspoof 5**
CMPE 258 – Deep Learning, Fall 2026

## Team

- Shriya Chockalingam
- Zahira Ghazali
- Smitha Soman

## Abstract

Modern text-to-speech (TTS) and voice-conversion (VC) systems produce speech that anti-spoofing detectors often miss when the attack algorithm was not seen in training. A deployed detector also faces shifts that have nothing to do with the generator: codecs and phone channels, and spoofs that have been deliberately post-processed to evade detection. A single pooled EER hides which of these is doing the damage.

ASVspoof 5 is the first public benchmark that labels all three shifts on one evaluation set, so their effects can be separated. This project uses it to ask **which distribution shifts actually break modern audio-deepfake detectors, and whether SSL-based detectors fail differently from older anti-spoofing architectures.** We compare the released ASVspoof 5 AASIST baseline with a frozen XLS-R front end plus a lightweight head. We score both on matched slices for generator, codec, and adversarial shift, and on DFADD as a test against recent diffusion and flow-matching TTS systems. Calibration and selective prediction are used as a diagnostic, to see whether each detector's errors are concentrated at low confidence or whether it is confidently wrong.

We do not claim per-condition reporting or calibration as new. The ASVspoof 5 organizers already report both for submitted systems, and Dao et al. (2026) already give a per-codec breakdown on ASVspoof 5. What we add is narrower: a non-SSL reference point, utterance-level pairing of coded and uncoded audio, adversarial conditions inside the same decomposition, calibration and selective prediction per shift type, and a control that separates codec effects from non-speech shortcuts.

## Research Questions

**Main question.** Which shifts break modern detectors: unseen generator families, codecs and compression, or adversarial post-processing? And do SSL-based detectors fail differently from older architectures?

- **RQ1 – Where.** For each detector, how far do minDCF and EER degrade under each shift, relative to a matched reference?
- **RQ2 – Architecture.** Is the ranking of shifts the same for AASIST and for the SSL detector, or does an SSL front end remove one failure mode while leaving another intact?
- **RQ3 – Confidence.** Under each shift, are errors concentrated at low confidence, so abstention helps, or is the detector confidently wrong?

## Problem Formulation

- **Task:** binary bona fide / spoof classification of a 16 kHz waveform, with an abstention option.
- **Input:** variable-length waveform, 4 s crops.
- **Output:** spoof probability, calibrated confidence, decide/abstain flag.
- **Constraints:** public datasets only; a single consumer GPU; SSL front ends frozen with features cached once; CPU latency of a quantized model reported, not targeted.

## Success Criteria

These are targets, not guaranteed outcomes.

1. Our evaluation code reproduces the official metrics of the released ASVspoof 5 AASIST baseline (target: Oct 12).
2. A shift-decomposed results table covering both detectors, every slice, and DFADD, with speaker-level bootstrap intervals (target: Nov 10).
3. An evidence-backed answer to RQ3 for each shift, separating calibration failures from ranking failures (target: Nov 10).

A null result on RQ2 is still a result.

## Data

**Primary: ASVspoof 5, Track 1.** Crowdsourced LibriVox speech (MLS English) in non-studio conditions, with 32 attacks.

| Partition | Attacks | Files | Use |
|---|---|---|---|
| Train | A01–A08 (zero-shot TTS) | 182,357 | Training; 10% of speakers held out as the in-distribution reference |
| Development | A09–A16 (unseen TTS and VC) | 140,950 | Model selection and calibration fitting |
| Evaluation | A17–A32 (new TTS/VC, 7 adversarial attacks, codecs C00–C11) | 680,774 | Testing only |

Codecs appear only in the evaluation partition, so calibration is always fitted on clean audio. The held-out reference is 40 training speakers (seed 258, stratified by gender). The released AASIST checkpoint was trained on all training speakers, so this reference is optimistic for AASIST. The speaker-disjoint comparison applies to the SSL detector, which is trained on the remaining 90%.

**Shift slices.** Each shift is measured against a matched reference, so its effect is isolated rather than pooled.

| Shift | Slice | Matched reference |
|---|---|---|
| Generator | Non-adversarial evaluation attacks at C00, grouped by family: legacy unit selection (A19), zero-shot TTS (A17, A21, A22, A28, A29), VC including diffusion VC (A24–A26) | Held-out training speakers, A01–A08, no codec |
| Codec | The same utterances under C01–C11, grouped as wideband DSP, neural (Encodec, MP3+Encodec), narrow-band DSP, and PSTN simulation | The same utterance at C00, linked through the protocol's `CODEC_SEED` column |
| Adversarial | Malafide, aimed at the countermeasure: A18 and A30 (from A17), A20 (from A12), A23 (from A09). Malacopula, aimed at ASV: A27, A31, A32 | The source attack at the same codec condition; reported as attack transfer, since Malafide was optimized against an AASIST countermeasure |
| Modern generators | DFADD, per generator | ASVspoof 5 reference; bona fide scores analyzed separately |

**Open world: DFADD.** Five recent diffusion and flow-matching TTS systems. Generation family, recording conditions, and speakers all differ, so it is reported per generator and separately from the primary protocol. Its bona fide audio is VCTK rather than LibriVox, so the shift hits both classes. We report bona fide score distributions separately, so that rejecting real speech is not mistaken for missing spoofs.

**Historical baseline: ASVspoof 2019 LA.** Kept only to check our evaluation code against the released AASIST checkpoint (0.83% EER) and, optionally, as an "old training data" detector scored on the ASVspoof 5 slices.

**Preprocessing.** 4 s crops (the first 64,600 samples, with short clips repeated, matching the official AASIST code). No augmentation in the base condition, so ablation effects stay attributable.

## Detectors

1. **AASIST**: end-to-end spectro-temporal graph attention over raw waveforms, and the official ASVspoof 5 baseline. We use the released checkpoint; training our own does not fit the schedule.
2. **Frozen XLS-R 300M + lightweight head**: a layer-weighted head over cached per-layer statistics, trained with three seeds. XLS-R was pretrained on MLS, which shares LibriVox speakers with ASVspoof 5, so a LibriSpeech-only wav2vec 2.0 backbone serves as the overlap control if time allows.

The strongest system type we are not running is a fine-tuned SSL front end with an AASIST or Conformer back end, which is what leads ASVspoof 5. Our conclusions are therefore scoped to frozen-feature detectors unless partial fine-tuning of the top layers is completed.

## Experiments

1. **Shift-decomposed evaluation (core).** Both detectors scored on every slice and on DFADD, with degradation against the matched reference and speaker-level bootstrap intervals, plus the full attack × codec grid as a heatmap.
2. **Calibration and selective prediction (diagnostic for RQ3).** Temperature scaling fitted only on the development set and frozen before evaluation, with maximum softmax probability (MSP) as the confidence baseline. Calibration (Cllr − minCllr, actDCF − minDCF) is reported separately from ranking (excess AURC, and how well confidence predicts errors).
3. **INT8 post-training quantization (optional).** Model size, CPU latency, and ΔEER per slice, only after the core table exists.

**Optional experiments, in priority order:** partial fine-tuning of the SSL top layers; the LibriSpeech-only backbone control; the 2019 LA detector; per-shift layer probes; INT8; DSP-only codec augmentation with neural codecs held out.

## Controls

| Control | Expected outcome | What it catches |
|---|---|---|
| Random-score detector on every slice | EER ≈ 50%, AURC ≈ base risk | Bugs in scoring, bootstrap, or risk–coverage code |
| Released AASIST vs. official metrics | Our numbers match the official package | Harness and protocol-parsing errors |
| Near-transparent MP3, applied by us | Near-zero degradation | A "codec effect" that is really a pipeline or loudness artifact |
| 8 kHz downsampling, applied by us | Large degradation for every detector | Slices that cannot detect a shift we know exists |
| Dev-fitted calibrator on in-distribution data | actDCF − minDCF ≈ 0 | Calibration fitted or applied incorrectly |
| Silence-and-duration-only classifier | Near chance | How much of each slice trivial cues already explain |
| VAD-trimmed rerun of the codec slice | Codec effect survives trimming | A non-speech shortcut masquerading as a codec effect |

**Falsification conditions, fixed in advance.** The contribution is unnecessary if (1) all detectors degrade in the same order with the same relative magnitude and overlapping intervals everywhere, in which case pooled EER reporting is adequate; or (2) the codec effect disappears under the VAD-trimmed control, in which case we are replicating a known non-speech shortcut and will report it as such.

## Evaluation Metrics

- **Discrimination:** minDCF (ASVspoof 5 primary metric), EER, AUROC.
- **Per shift:** results for every generator, codec, and adversarial slice; the attack × codec heatmap; DFADD EER per generator.
- **Calibration:** Cllr, actDCF, ECE, reliability diagrams.
- **Abstention:** risk–coverage curves, AURC, excess AURC, error rate at 80% and 90% coverage.
- **Deployment (optional):** latency, model size, ΔEER.

All results carry 95% speaker-level bootstrap intervals, and a difference is claimed only when a paired bootstrap supports it.

## Current Status

| Item | Owner | Status |
|---|---|---|
| ASVspoof 5 protocols downloaded and checked (row counts match) | Shriya | Done |
| Released AASIST scored on dev, eval, and held-out training speakers | Shriya | Done |
| Official-metric reproduction (success criterion 1) | Shriya, Smitha | Done: full-dev EER 15.20%, matching the 15.2% reported in the official repo |
| 10% held-out speaker split (seed 258) | Shriya | Done |
| DFADD + VCTK test set prepared and scored | Shriya | In progress |
| XLS-R feature caching and SSL head (3 seeds) | Zahira | _update_ |
| Evaluation harness: slices, codec/adversarial pairing, bootstrap | Smitha | _update_ |
| Controls, calibration, risk–coverage | Smitha, Zahira | Not started |

**Released AASIST, official ASVspoof 5 checkpoint (our scores, official evaluation package):**

| Set | Files | EER (%) | minDCF |
|---|---|---|---|
| Dev, full | 140,950 | 15.20 | 0.3148 |
| Eval, pooled | 680,774 | 28.11 | 0.6663 |
| Held-out training speakers* | 18,583 | 0.05 | 0.0010 |

\*Not speaker-disjoint for AASIST, since the released checkpoint was trained on these speakers. actDCF is about 0.97 on every set because the scores are uncalibrated logits. Calibrated results will be reported once temperature scaling is fitted on dev.

## Team Task Split

| Member | Owns |
|---|---|
| **Shriya Chockalingam** | Dataset pipeline and preprocessing; AASIST scoring from the released checkpoint; DFADD and VCTK preparation; main evaluation runs and ablations |
| **Zahira Ghazali** | SSL detector (frozen XLS-R + lightweight head); calibration and selective prediction; ECE, reliability diagrams, and risk–coverage analysis; layer probes and the backbone control if time allows |
| **Smitha Soman** | Shift-decomposed evaluation harness (protocol parsing, slice construction, codec and adversarial pairing, speaker-level bootstrap, all results tables and figures); DFADD cross-dataset evaluation; control experiments; INT8 quantization and latency benchmarking; demo application and video |

All three share the final report and presentation. We meet weekly to keep splits, seeds, and crop logic identical across runs.

## Timeline

| Week | Dates | Work | Milestone |
|---|---|---|---|
| 1 | Sep 22–28 | Host machine; protocols; slices and scoring wrapper; audio download | Slice counts checked |
| 2 | Sep 29–Oct 5 | Audio extracted; AASIST scores dev and eval; pilot; random-score, transparent-MP3, and 8 kHz controls | Pilot decides cell sizes |
| 3 | Oct 6–12 | Harness reproduces official AASIST metrics; XLS-R feature caching begins | Hard checkpoint |
| 4–5 | Oct 13–26 | SSL head trained (3 seeds); first shift table for generator and codec slices | Progress check |
| 6–7 | Oct 27–Nov 9 | DFADD; calibration and risk–coverage per slice; adversarial slice; VAD-trimmed and silence-only controls; demo skeleton | Results freeze Nov 10 |
| 8–9 | Nov 10–23 | Optional experiments only if the core is complete; figures; slides; backup demo video | Full draft deck |
| 10 | Nov 24–Dec 1 | Rehearsal and buffer | Demo and presentation |

## Repository Layout

```
data/          # protocol parsing, speaker split, DFADD/VCTK preparation
baselines/     # AASIST scoring and the SSL detector (feature caching, head training)
eval/          # shift slices, codec/adversarial pairing, bootstrap, official metrics
calibration/   # temperature scaling, selective prediction, risk–coverage
controls/      # random-score, MP3, 8 kHz, silence-only, VAD-trimmed controls
deployment/    # INT8 conversion and CPU latency (optional)
notebooks/     # Colab notebooks used to produce scores
results/       # tables, heatmaps, reliability and risk–coverage plots
docs/          # proposal, literature survey, AI critique
```

## Reproducibility Rules

- Train, dev, and eval partitions follow the official ASVspoof 5 Track 1 protocol and are never mixed.
- The held-out speaker split is created once (seed 258) and reused by everyone.
- Calibration and thresholds are fitted only on dev and frozen before any evaluation-set scoring.
- Score files use the official format (`filename`, `cm-score` = bona fide logit, higher means more likely real) and are scored with the official evaluation package.
- Record seeds, crop logic, model configuration, and checkpoint identifiers for every run.
- Published numbers are labeled as reference values; only numbers produced by our code are labeled as results.
- Experiments that are not completed are reported as not run.

## Further Reading

- [`LITERATURE_SURVEY.md`](LITERATURE_SURVEY.md)
- [`AI_CRITIQUE.md`](AI_CRITIQUE.md)
