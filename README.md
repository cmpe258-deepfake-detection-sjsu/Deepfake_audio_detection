# What Breaks Modern Audio Deepfake Detectors?

**Generator, codec, and adversarial shift on ASVspoof 5**

CMPE 258 – Deep Learning, Fall 2026

## Team

Shriya Chockalingam
Zahira Ghazali
Smitha Soman

## Abstract

Modern text-to-speech (TTS) and voice-conversion (VC) systems produce speech that anti-spoofing detectors often miss when the attack algorithm was not seen in training. A deployed detector also faces shifts that have nothing to do with the generator: codecs and phone channels, and spoofs that have been deliberately post-processed to evade detection. A single pooled EER hides which of these is doing the damage.

ASVspoof 5 is the first public benchmark that labels all three shifts on one evaluation set, so their effects can be separated. This project uses it to ask which distribution shifts actually break modern audio-deepfake detectors, and whether SSL-based detectors fail differently from older anti-spoofing architectures. We compare the released ASVspoof 5 AASIST baseline with a frozen XLS-R front end plus a lightweight head. We score both on matched slices for codec and adversarial shift, report generator performance as between-family variation, and evaluate on DFADD as a test against recent diffusion and flow-matching TTS systems. Calibration and selective prediction are used as a diagnostic, to see whether each detector's errors are concentrated at low confidence or whether it is confidently wrong.

We do not claim per-condition reporting or calibration as new. The ASVspoof 5 organizers already report both for submitted systems, and Dao et al. (Odyssey 2026) give a per-codec breakdown on ASVspoof 5. What we add is narrower: a non-SSL reference point, utterance-level pairing of coded and uncoded audio so that the codec effect can be split by class, adversarial conditions inside the same decomposition, calibration and selective prediction per shift type, and a control that separates codec effects from non-speech shortcuts.

## Research Questions

**Main question.** Which shifts break modern detectors: unseen generator families, codecs and compression, or adversarial post-processing? And do SSL-based detectors fail differently from older architectures?

- **RQ1 – Where.** For each detector, how far does EER degrade under each matched shift, relative to a reference that differs only in that shift? Generator performance is reported as variation across families, not as degradation (see *A note on references*).
- **RQ2 – Architecture.** Do the two detectors degrade in the same order under matched contrasts, or does an SSL front end remove one failure mode while leaving another intact?
- **RQ3 – Confidence.** Under each shift, are errors concentrated at low confidence, so abstention helps, or is the detector confidently wrong?

## A note on references

Not every shift axis admits a matched comparison on this benchmark, and conflating them would overstate what the numbers support.

**Codec and adversarial are matched.** A coded file can be compared against the very same source utterance uncoded, because `CODEC_SEED` names that source file (verified: all 171,602 coded seeds resolve to an uncoded filename, with speaker and attack agreeing in 100% of 509,172 coded files). A18 and A30 are Malafide filters applied to A17, so comparing them against A17 holds generator, partition and codec fixed. Both are reported as a difference in EER points with a speaker bootstrap on the difference, not on each EER separately.

**Generator is not matched.** Measuring degradation from seen to unseen generators would need A01–A08 as a baseline, and that exists only in the training partition. The released AASIST checkpoint was trained on every training speaker, so the held-out split is not a valid reference for it; and for the SSL detector the comparison still crosses partitions, changing speakers and recording conditions along with generators. Generator results are therefore reported as between-family variation at a fixed codec condition, against the uncoded bona fide pool from the same partition.

Because of this, we do not rank the three shift axes against one another. We compare matched contrasts to matched contrasts.

## Problem Formulation

**Task:** binary bona fide / spoof classification of a 16 kHz waveform, with an abstention option.
**Input:** variable-length waveform, 4 s crops.
**Output:** spoof probability, calibrated confidence, decide/abstain flag.
**Constraints:** public datasets only; a single consumer GPU; SSL front ends frozen with features cached once; CPU latency of a quantized model reported, not targeted.

## Success Criteria

These are targets, not guaranteed outcomes.

- Our evaluation code reproduces the official metrics of the released ASVspoof 5 AASIST baseline. **Met:** full-dev EER 15.20% against the 15.2% reported in the official repo, and our own EER implementation reproduces the official evaluation package exactly on pooled eval (28.11% both ways).
- A results table covering both detectors, every matched contrast, the generator families and DFADD, with speaker-level bootstrap intervals (target: Nov 10).
- An evidence-backed answer to RQ3 for each shift, separating calibration failures from ranking failures (target: Nov 10).
- A null result on RQ2 is still a result.

## Data

**Primary: ASVspoof 5, Track 1.** Crowdsourced LibriVox speech (MLS English) in non-studio conditions, with 32 attacks.

| Partition | Attacks | Files | Use |
|---|---|---|---|
| Train | A01–A08 (zero-shot TTS) | 182,357 | Training the SSL head; 10% of speakers held out |
| Development | A09–A16 (unseen TTS and VC) | 140,950 | Model selection and calibration fitting |
| Evaluation | A17–A32 (new TTS/VC, 7 adversarial attacks, codecs C01–C11) | 680,774 | Testing only |

Codecs appear only in the evaluation partition, so calibration is always fitted on clean audio. The uncoded condition is labelled `-` in the protocol, not `C00`.

The held-out reference is 40 training speakers (seed 258, stratified by gender). It is **not** used as AASIST's reference — the released checkpoint was trained on all training speakers, which is why it scores 0.05% EER there. It remains the correct in-distribution reference for the SSL detector, which never sees those speakers.

**Shift slices.**

| Shift | Slice | Reference | Matched? |
|---|---|---|---|
| Codec | The same source utterances under C01–C11. Grouping from the ASVspoof 5 paper: 16 kHz DSP (C01 opus, C02 amr, C03 speex, C05 mp3, C06 m4a), neural (C04 Encodec, C07 mp3+Encodec), 8 kHz DSP (C08 opus, C09 amr, C10 speex), mixed 8 kHz (C11) | The same utterance uncoded, linked through `CODEC_SEED`. 33,757 source utterances appear under all eleven codecs. | Yes — utterance-level |
| Adversarial | Malafide, aimed at the countermeasure: A18 and A30 (from A17), A20 (from A12), A23 (from A09). Malacopula, aimed at ASV: A27, A31, A32 | The source attack, uncoded. Only A18 and A30 admit a matched contrast here; A20 and A23 derive from dev attacks, so their sources are not in the evaluation partition. | Yes — for A18 and A30 |
| Generator | Non-adversarial evaluation attacks, uncoded, grouped by family: legacy unit selection (A19), zero-shot TTS (A17, A21, A22, A28, A29), VC including diffusion VC (A24–A26) | The uncoded bona fide pool from the same partition | No — variation only |
| Modern generators | DFADD, per generator | ASVspoof 5 reference; bona fide scores analyzed separately | No — cross-dataset |

Malafide was optimised against an AASIST countermeasure, so for the SSL detector these are transfer attacks. Any cross-architecture statement about adversarial robustness is reported as attack transfer, not as a property of the architecture.

**Open world: DFADD.** Five recent diffusion and flow-matching TTS systems, 3,755 utterances. Generation family, recording conditions, and speakers all differ, so it is reported per generator and separately from the primary protocol — separately because its bona fide audio is VCTK rather than LibriVox, not because it matters less. We report bona fide score distributions separately, so that rejecting real speech is not mistaken for missing spoofs.

**Historical baseline: ASVspoof 2019 LA.** Kept only to check our evaluation code against the released AASIST checkpoint (0.83% EER).

**Preprocessing.** 4 s crops (the first 64,600 samples, with short clips repeated, matching the official AASIST code). No augmentation in the base condition, so ablation effects stay attributable.

## Detectors

- **AASIST:** end-to-end spectro-temporal graph attention over raw waveforms, and the official ASVspoof 5 baseline. We use the released checkpoint; training our own does not fit the schedule.
- **Frozen XLS-R 300M + lightweight head:** a layer-weighted head over cached per-layer statistics, trained with three seeds. XLS-R was pretrained on MLS, which shares LibriVox speech with ASVspoof 5.

The two systems differ in front end, back end, training recipe and training subset, so differences between them are system-level, not isolated architectural causes. The strongest system type we are not running is a fine-tuned SSL front end with an AASIST or Conformer back end, which is what leads ASVspoof 5. Our conclusions are scoped to frozen-feature detectors.

All three seeds are reported alongside the seed-mean ensemble, so that no seed is selected after seeing results. On pooled eval the seeds span 7.272–7.663% and the ensemble lands at 7.435%, inside that range — ensembling did not improve EER here.

## Experiments

**Matched shift contrasts (core).** Codec and adversarial degradations for both detectors, each against a reference differing only in that shift, with a speaker bootstrap on the difference. Generator families reported as variation. Plus the full attack × codec grid as a heatmap.

**Paired per-utterance codec analysis.** For each coded file, the score change against its own uncoded original, split by class — which separates "the codec makes fakes look real" from "the codec makes real speech look fake".

**Calibration and selective prediction (diagnostic for RQ3).** Temperature scaling fitted only on the development set and frozen before evaluation, with maximum softmax probability (MSP) as the confidence baseline. Reported per shift type, with coverage broken out by class.

> Note: for binary logits, MSP confidence is monotonic in |z₁ − z₀|, so positive scalar temperature scaling preserves the confidence ordering exactly and cannot change a risk–coverage curve at fixed coverage. Calibration metrics (ECE, Cllr, actDCF) and ranking metrics (AURC, excess AURC) are therefore reported separately and answer independent questions.

**INT8 post-training quantization (optional).** Model size, CPU latency, and ΔEER per slice, only after the core table exists.

Optional experiments, in priority order: partial fine-tuning of the SSL top layers; a different SSL backbone as a sensitivity analysis on pretraining overlap (note that LibriSpeech is also LibriVox-derived, so it is not an overlap-free control); the 2019 LA detector; per-shift layer probes; INT8.

## Controls

| Control | Expectation | What it catches |
|---|---|---|
| Random-score detector on every slice | EER near 50% within sampling variation | Bugs in scoring, bootstrap, or risk–coverage code |
| Released AASIST vs. official metrics | Our numbers match the official package | Harness and protocol-parsing errors. **Met:** pooled eval 28.11% both ways |
| Near-transparent MP3 | Little or no degradation | A "codec effect" that is really a pipeline or loudness artifact |
| 8 kHz downsampling | A shift we know exists should be detectable by the slice machinery | Slices too coarse to register a real condition change |
| Silence-and-duration-only classifier | Near chance | How much of each slice trivial cues already explain |
| VAD-trimmed rerun of the codec slice | Codec effect survives trimming, measured on utterances whose trim boundaries agree across conditions | A non-speech shortcut masquerading as a codec effect |

**Falsification conditions, fixed in advance.** The contribution is weakened if (1) both detectors degrade in the same order under matched contrasts with intervals that do not separate them — in which case we report that the architectures behave alike, rather than treating overlapping intervals as proof of equivalence; or (2) the codec effect disappears under the VAD-trimmed control, in which case we are replicating a known non-speech shortcut and will report it as such. Note that trimming also changes duration and how the fixed-length crop repeats short clips, so a disappearing effect shows sensitivity to trimming rather than proving silence is the cause.

Condition (2) is assessed on a boundary-matched comparison, for the reason given below.

### Control: leading and trailing silence (VAD)

**Design.** 5,000 uncoded source utterances and their C04/C07/C08/C10 versions were rescored after trimming leading and trailing silence. The statistic is a difference-in-differences, because the question is whether trimming changed the *codec effect*, not whether EER moved:

```
effect_untrimmed = EER(codec, untrimmed) - EER(uncoded, untrimmed)
effect_trimmed   = EER(codec, trimmed)   - EER(uncoded, trimmed)
DiD              = effect_trimmed - effect_untrimmed
```

All four EERs come from the same speaker draw, 1,000 draws, speakers resampled rather than utterances. An interval spanning zero means trimming did not detectably change the codec effect. An interval clearly below zero would be the falsification case.

**The VAD ran per condition, so the naive comparison is confounded.** The VAD was applied independently to each coded version rather than once to the uncoded original with the boundaries reused. On the two 8 kHz conditions it places speech onset about one analysis frame later:

| Condition | Lead silence vs uncoded (median) | (mean) | Trail silence vs uncoded (mean) |
|---|---|---|---|
| C04 | 0.000 s | 0.0005 s | −0.005 s |
| C07 | 0.000 s | 0.0008 s | −0.005 s |
| C08 | **0.032 s** | 0.082 s | **0.133 s** |
| C10 | **0.032 s** | 0.062 s | **0.106 s** |

For C08 and C10 the trimmed file is therefore not the uncoded file minus silence; it is that minus roughly 200 ms of additional speech. Both detectors are unstable on sub-100 ms windows, so the extra cut raises EER and inflates the DiD. C04 and C07 agree with the uncoded original to under a millisecond and are a valid test as run.

**Boundary-matched subset.** The reported control restricts to source utterances whose VAD boundaries agree across all five conditions within a tolerance, and runs the same statistic on the existing scores. Agreement is sharply bimodal — 17.8% match within 5 ms and the figure is unchanged at 20 ms, then rises to 55.4% at 50 ms — consistent with a one-frame quantised disagreement. The 50 ms subset (2,718 utterances, 682 speakers) is the headline.

XLS-R, DiD in EER points, 95% CI:

| Codec | All scored (confounded) | Boundary-matched ±50 ms |
|---|---|---|
| C04 | +0.61 [−0.04, 1.12] | +0.04 [−0.83, 0.73] |
| C07 | −0.18 [−0.73, 0.53] | −0.47 [−1.28, 0.35] |
| C08 | **+1.37 [0.54, 1.94]** | 0.00 [−0.62, 0.71] |
| C10 | **+1.90 [1.32, 2.56]** | +0.18 [−0.37, 0.86] |

AASIST, DiD in EER points, 95% CI:

| Codec | All scored | Boundary-matched ±50 ms |
|---|---|---|
| C04 | −0.36 [−1.07, 0.61] | +0.70 [−0.79, 1.65] |
| C07 | −0.08 [−0.89, 0.83] | +0.95 [−0.86, 1.65] |
| C08 | −0.04 [−0.88, 0.89] | +1.03 [−0.58, 1.93] |
| C10 | +0.28 [−0.69, 1.17] | +0.66 [−0.94, 1.45] |

**Result.** On the boundary-matched comparison every interval spans zero for both detectors, so the codec effect is not explained by leading or trailing silence. Falsification condition (2) is not met.

The two XLS-R conditions that appeared to change on the full set do not survive matching, and the matched intervals exclude the confounded estimates rather than merely failing to reproduce them (C10: upper bound 0.86 against a point estimate of 1.90, on three times the data with narrower intervals). That is the confound being removed, not a smaller sample losing power.

AASIST's full-set numbers were never affected. At ~22% uncoded EER an extra frame off the onset does not move it, whereas XLS-R at ~2% had room to degrade; the boundary problem was only visible on the more accurate detector.

**Caveats, stated.**

- The matched subset is **selected, not random**. Utterances whose VAD agrees under an 8 kHz codec are plausibly the ones the codec damaged least, and the codec effects themselves differ on the subset (C08's XLS-R effect is 6.51 points there against 8.34 on the full set). The matched result is evidence about *direction* — the effect is not silence-driven — and not a revised estimate of effect size. Effect sizes are reported from the full eval set.
- All four of AASIST's matched point estimates are positive at both tolerances, none individually resolved. The offset is uniform across codecs, including the two whose boundaries always matched, and traces to the uncoded condition improving more under trimming (18.29% → 17.77%) than the coded conditions did. Trimming slightly helps AASIST on clean audio and not on degraded audio.
- A clean fix would re-trim every file using the uncoded boundaries and rescore. Not run: it requires regenerating the trimmed audio and a full XLS-R rescore, and the matched-subset analysis answers the question the control was asked.

Reproduce with `eval/vad_matched_boundary.py` (set `MODEL` and `TOL_S`). Outputs: `results/258_vad_matched_xlsr_50ms.csv`, `results/258_vad_matched_aasist_50ms.csv`, `results/258_vad_matched_aasist_20ms.csv`. The unmatched run is `eval/vad_bootstrap.py` → `results/258_vad_bootstrap.csv`.

## Evaluation Metrics

- **Discrimination:** minDCF (ASVspoof 5 primary metric), EER, AUROC.
- **Matched contrasts:** difference in EER points against a reference differing only in the shift, with a 95% speaker bootstrap on the difference.
- **Per condition:** every codec and adversarial slice; the attack × codec heatmap; DFADD EER per generator.
- **Calibration:** Cllr, actDCF, ECE, reliability diagrams.
- **Abstention:** risk–coverage curves, AURC, excess AURC, error rate at 80% and 90% coverage, with coverage reported separately for bona fide and spoof.
- **Deployment (optional):** latency, model size, ΔEER.

A difference is claimed only when a paired bootstrap on that difference supports it.

## Current Status

| Item | Owner | Status |
|---|---|---|
| ASVspoof 5 protocols downloaded and checked | Shriya | Done |
| Released AASIST scored on dev, eval and held-out training speakers | Shriya | Done |
| Official-metric reproduction | Shriya, Smitha | Done — full-dev EER 15.20% |
| 10% held-out speaker split (seed 258) | Shriya | Done |
| DFADD prepared and scored | Shriya | Done — 3,755 utterances, per-generator metrics |
| XLS-R feature caching, train/dev/eval | Zahira | Done |
| Three-seed head trained; dev and eval scores | Zahira | Done — seed spread 0.391 points on pooled eval |
| Temperature scaling fitted on dev | Zahira | Done — T = 1.5223; eval ECE 0.0330 → 0.0268 |
| Evaluation harness: protocol parsing, slices, bootstrap | Smitha | Done |
| Codec pairing resolved via `CODEC_SEED` | Smitha | Done — 100% of coded files linked to their source |
| Matched contrasts, both detectors, paired bootstraps | Smitha | Done |
| Paired per-utterance codec deltas by class | Smitha | Done |
| Attack × codec heatmap | Smitha | Done |
| VAD trimming and rescoring, AASIST | Shriya | Done — 5,000 source utterances, four codecs plus uncoded |
| VAD rescoring, XLS-R | Zahira | Done — same 93-original exclusion across all five conditions |
| VAD paired bootstrap and boundary-matched control | Smitha | Done — both detectors; per-condition VAD boundaries found and controlled for |
| Per-slice calibration and risk–coverage | Zahira | Done — per-slice selective prediction for both detectors; AASIST temperature fit has no finite optimum, so no calibrated AASIST probabilities are claimed |
| DFADD folded into the main table | Smitha | Not started |
| Random-score and silence-only controls | Smitha, Shriya | Not started |
| Rates at a fixed dev-selected threshold | Smitha | Not started |

## Results so far

**The matched contrasts order oppositely across the two detectors.**

| Contrast | AASIST | XLS-R |
|---|---|---|
| A17 → A18 (adversarial, generator held fixed) | +43.28 [41.84, 44.72] | +2.22 [1.91, 2.58] |
| uncoded → C07 (mp3+Encodec) | +7.41 [6.86, 7.93] | +16.49 [15.86, 17.19] |

Both are paired degradations with intervals on the difference. Adversarial post-processing costs AASIST 5.8× what C07 does; C07 costs XLS-R 7.4× what the adversarial step does.

**AASIST is sign-inverted on vocoder-free synthesis; XLS-R is not.** A19 and A12 are the only two waveform-concatenation attacks in the benchmark, and the only two AASIST scores as more bona fide than real speech — AUC 0.347 on eval and 0.194 on dev, the latter holding for 99.2% of 386 speakers. XLS-R inverts on nothing. The inversion is codec-independent: A19 stays between 53.9% and 61.2% EER across all twelve conditions.

**Codec degradation is concentrated in the bona fide class.** Paired per-utterance deltas are negative for real speech under every codec and both detectors; spoof scores move far less. For XLS-R the two neural-codec conditions produce the largest bona fide drops in the table.

**The codec effect survives the VAD control on both detectors.** On the boundary-matched subset every difference-in-differences interval spans zero. The two XLS-R conditions that appeared to change on the unmatched comparison were an artefact of the VAD cutting about 200 ms more from the 8 kHz files than from their uncoded sources; see *Control: leading and trailing silence*. Attack slices are likewise restricted to uncoded examples, because pooling codec conditions into a generator slice moves A19's AUC anywhere from 0.23 to 0.52 depending on the mix.

**Independent agreement on the codec ordering.** Dao et al. (Odyssey 2026) report C04 and C07 as the two worst conditions for a different SSL system, with C07 worse than C04 — the same ordering. Their analysis is pooled per-condition EER, not the paired class-specific decomposition, so this is agreement on the ordering rather than a replication of our mechanism.

## Team Task Split

| Member | Owns |
|---|---|
| Shriya Chockalingam | Dataset pipeline and preprocessing; AASIST scoring from the released checkpoint; DFADD and VCTK preparation; audio-dependent controls |
| Zahira Ghazali | SSL detector (frozen XLS-R + lightweight head); calibration and selective prediction; ECE, reliability diagrams, and risk–coverage analysis; layer probes and backbone sensitivity if time allows |
| Smitha Soman | Shift-decomposition evaluation harness (protocol parsing, slice construction, codec and adversarial pairing, speaker-level bootstrap, all results tables and figures); DFADD cross-dataset evaluation; control experiments; INT8 quantization and latency benchmarking; demo application and video |

All three share the final report and presentation. We meet weekly to keep splits, seeds, and crop logic identical across runs.

## Timeline

| Week | Dates | Work | Milestone |
|---|---|---|---|
| 1 | Sep 22–28 | Host machine; protocols; slices and scoring wrapper; audio download | Slice counts checked |
| 2 | Sep 29–Oct 5 | Audio extracted; AASIST scores dev and eval; pilot | Pilot decides cell sizes |
| 3 | Oct 6–12 | Harness reproduces official AASIST metrics; XLS-R feature caching begins | Hard checkpoint |
| 4–5 | Oct 13–26 | SSL head trained (3 seeds); first shift table | Progress check |
| 6–7 | Oct 27–Nov 9 | DFADD; calibration and risk–coverage per slice; adversarial slice; VAD-trimmed and silence-only controls; demo skeleton | Results freeze Nov 10 |
| 8–9 | Nov 10–23 | Optional experiments only if the core is complete; figures; slides; backup demo video | Full draft deck |
| 10 | Nov 24–Dec 1 | Rehearsal and buffer | Demo and presentation |

## Repository Layout

```
data/          # protocol parsing, speaker split, DFADD/VCTK preparation
baselines/     # AASIST scoring and the SSL detector (feature caching, head training)
eval/          # shift slices, codec/adversarial pairing, bootstrap, official metrics,
               #   and the control scripts that run on score files
calibration/   # temperature scaling, selective prediction, risk–coverage
controls/      # audio-dependent controls (MP3, 8 kHz, silence-only)
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
- Generator and attack slices are computed on uncoded examples only (`CODEC == '-'`), against the uncoded bona fide pool. Pooling codec conditions into an attack slice mixes codec shift into a number meant to isolate generator shift.
- Every reported number maps to a script, a score-file version, a checkpoint identifier and a configuration. The same speaker bootstrap draws are used across systems and paired conditions.
- Seed variation and speaker bootstrap intervals measure different sources of uncertainty and are reported separately.
- Published numbers are labeled as reference values; only numbers produced by our code are labeled as results.
- Experiments that are not completed are reported as not run.

## Further Reading

- [LITERATURE_SURVEY.md](LITERATURE_SURVEY.md)
- [AI_CRITIQUE.md](AI_CRITIQUE.md)
