# Generalizable Audio Deepfake Detection Under Unseen Attacks

**CMPE 258 \- Deep Learning, Fall 2026** 

## Team

- Shriya Chockalingam  
- Zahira Ghazali  
- Smitha Soman

## Abstract

Modern text-to-speech (TTS) and voice-conversion (VC) systems can generate highly convincing speech, while anti-spoofing detectors may lose performance when evaluated on synthesis methods that were not represented during training. This project studies whether a detector trained on known spoofing attacks can generalize to unseen attacks **and provide reliable confidence estimates that support abstention when the model is uncertain**.

Our primary benchmark is ASVspoof 2019 LA. Following the official protocol, training and development data use attacks A01–A06, while A07–A19 form the official evaluation set. We additionally perform a controlled leave-one-attack-out rotation within A01–A06 to measure transfer to an unseen attack while keeping the official evaluation partition untouched. ASVspoof 2021 DF is used for codec-shift robustness, and DFADD is used as a secondary cross-dataset test against modern diffusion/flow-matching TTS systems.

We reproduce two credible baselines: AASIST and a self-supervised-learning (SSL) front end with a lightweight classifier. The project then evaluates two controlled extensions: **attack-aware training**, which balances training batches across known attack types, and **post-hoc calibration with selective prediction**, where temperature scaling is fitted only on the development set and the detector can abstain below a confidence threshold. Selective prediction is evaluated against a maximum-softmax-probability (MSP) confidence baseline using risk-coverage curves and AURC. Finally, the selected model is converted to INT8 and evaluated for CPU latency, model size, and EER degradation.

The project does not claim that unseen-attack generalization, calibration, or attack balancing is individually new. Instead, the research question is whether this combination provides a measurable improvement in **generalization-aware and uncertainty-aware evaluation** under unseen attacks while remaining feasible for deployment on consumer hardware.

## Problem Formulation

**Input:** 16 kHz speech waveform, using 4-second crops.

**Output:**

- spoof probability  
- confidence score  
- decision: bona fide / spoof / abstain

**Primary research question:** Can a detector trained on known spoofing attacks maintain useful performance on unseen attacks while producing confidence estimates that enable beneficial selective prediction?

## Evaluation Targets

Targets are feasibility goals, not guaranteed outcomes:

- Unseen-attack EER: within 2× of in-domain EER on designated headline attacks  
- Lower AURC than an MSP baseline at matched coverage  
- Report ECE and reliability diagrams for calibration  
- INT8 CPU inference under 100 ms per 4-second crop  
- Absolute EER degradation after quantization ≤ 1 percentage point

## Data

1. **ASVspoof 2019 LA — primary benchmark**  
     
   - Official training/development protocol  
   - A01–A06 used for training/development  
   - A07–A19 used as the official unseen evaluation set  
   - 3–4 headline attacks selected from different synthesis families

   

2. **ASVspoof 2021 DF — codec robustness**  
     
   - Use official codec conditions rather than creating a new compression protocol

   

3. **DFADD — secondary open-world evaluation**  
     
   - 3,755 utterances  
   - Five modern diffusion/flow-matching TTS systems  
   - Used as a cross-dataset test, reported separately from the primary ASVspoof protocol

## Baselines

- **AASIST:** established end-to-end anti-spoofing baseline  
- **SSL \+ lightweight classifier:** WavLM or XLS-R representation with a lightweight binary head

Spectro-temporal fusion is treated as an established baseline capability rather than as a project contribution.

## Proposed Experiments

### 1\. Attack-aware training

Compare conventional random sampling against batches balanced across A01–A06.

Additional controlled experiment:

- train on five attacks  
- evaluate on the sixth  
- rotate across A01–A06

### 2\. Calibration and selective prediction

- Fit temperature scaling using the development set only  
- Freeze calibration before unseen-attack evaluation  
- Compare calibrated confidence with maximum softmax probability  
- Sweep coverage/confidence thresholds  
- Report risk-coverage curves, AURC, ECE, and reliability diagrams

### 3\. Deployment

- Post-training INT8 quantization  
- Measure model size, CPU latency, EER, and absolute ΔEER

## Success Criteria

The project is successful if it produces a reproducible experimental comparison that answers:

1. Does attack-aware sampling improve transfer to unseen attacks?  
2. Does post-hoc calibration improve confidence quality?  
3. Does calibration produce better selective prediction than MSP at matched coverage?  
4. Do these benefits survive cross-dataset and codec shifts?  
5. Can the resulting model meet the deployment constraints?

A negative result is still meaningful if the experiments are controlled and reproducible.

## Team Task Split

| Member | Owns | Why |
| :---- | :---- | :---- |
| **Shriya Chockalingam** | Data pipeline and preprocessing; AASIST baseline reproduction; attack-aware training (balanced-batch sampling); main training runs and ablations | Foundational — blocks the other two roles, so it's scheduled first and needs to be the most solidly owned piece |
| **Zahira Ghazali** | SSL baseline (frozen front-end \+ lightweight head); calibration and selective prediction (temperature scaling, MSP comparison); ECE, reliability diagrams, risk-coverage/AURC analysis | This is the project's actual novelty claim — owning it end-to-end (implementation *and* the evaluation metrics that back the claim) keeps that story coherent under one person |
| **Smitha Soman** | Deployment & cross-dataset evaluation: INT8 post-training quantization, CPU latency/model-size benchmarking, ASVspoof 2021 DF codec-shift eval, DFADD cross-dataset eval, demo application and video, A01–A06 leave-one-attack-out rotation | Directly matches an embedded-systems/production background — this is the "does it actually run and hold up outside the lab" role, which is also the most portfolio-relevant piece for ML/MLOps engineering roles |

## Repository Layout

data/           \# dataset preparation, protocol handling, preprocessing

baselines/      \# AASIST and SSL baseline implementations/configs

models/         \# attack-aware training, calibration, selective prediction

eval/           \# EER/AUROC, ECE, reliability, risk-coverage/AURC

deployment/     \# INT8 conversion and CPU latency benchmarking

experiments/    \# experiment configs and run documentation

docs/           \# literature survey, AI audit, proposal materials

results/        \# tables, plots, and reproducibility artifacts

## Reproducibility Rules

- Keep train/dev/test partitions fixed once established.  
- Fit calibration parameters only on development data.  
- Never tune thresholds or calibration parameters on unseen evaluation attacks.  
- Record seeds, crop logic, preprocessing, model configuration, and checkpoint identifiers.  
- Distinguish published/reference numbers from results produced by our implementation.  
- Report incomplete experiments as not run rather than estimating results.

## References

See:

- `LITERATURE_SURVEY.md`  
- `AI_CRITIQUE.md`
