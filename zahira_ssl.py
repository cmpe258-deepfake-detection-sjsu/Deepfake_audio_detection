"""Local frozen-XLS-R detector and calibration pipeline for Zahira's project work."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import tarfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

SAMPLE_RATE = 16_000
CROP_SAMPLES = 64_600
PROTOCOL_COLUMNS = [
    "SPEAKER_ID",
    "FLAC_FILE_NAME",
    "SPEAKER_GENDER",
    "CODEC",
    "CODEC_Q",
    "CODEC_SEED",
    "ATTACK_TAG",
    "ATTACK_LABEL",
    "KEY",
    "TMP",
]
DEFAULT_MODEL = "facebook/wav2vec2-xls-r-300m"
DEFAULT_SEEDS = (258, 259, 260)


def project_paths(project_dir: Path) -> dict[str, Path]:
    project_dir = project_dir.resolve()
    return {
        "data": project_dir / "local_data" / "asvspoof5_data",
        "archives": project_dir / "local_data" / "work",
        "outputs": project_dir / "outputs" / "zahira_ssl",
    }


def load_protocol(split: str, paths: dict[str, Path]) -> pd.DataFrame:
    protocol_names = {
        "train": "ASVspoof5.train.tsv",
        "dev": "ASVspoof5.dev.track_1.tsv",
        "eval": "ASVspoof5.eval.track_1.tsv",
    }
    if split not in protocol_names:
        raise ValueError(f"Unknown split: {split}")
    protocol_path = paths["data"] / protocol_names[split]
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Protocol file not found: {protocol_path}")
    frame = pd.read_csv(
        protocol_path,
        sep=r"\s+",
        header=None,
        names=PROTOCOL_COLUMNS,
        dtype=str,
    )
    frame["filename"] = frame["FLAC_FILE_NAME"]
    if frame["filename"].duplicated().any():
        raise ValueError(f"Duplicate filenames in {protocol_path}")
    if not frame["KEY"].isin(["bonafide", "spoof"]).all():
        raise ValueError(f"Unexpected labels in {protocol_path}")

    if split == "train":
        split_path = paths["outputs"].parent / "shriya_aasist" / "splits" / "train_kept_speakers.txt"
        if not split_path.is_file():
            raise FileNotFoundError(
                f"Expected shared speaker split at {split_path}. "
                "Run the AASIST notebook through its held-out speaker split cell first."
            )
        kept_speakers = set(split_path.read_text(encoding="utf-8").split())
        frame = frame[frame["SPEAKER_ID"].isin(kept_speakers)].reset_index(drop=True)
        if frame.empty:
            raise ValueError("The kept-speaker split selected no training examples.")
    return frame


def prepare_waveform(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"Expected {SAMPLE_RATE} Hz audio, found {sample_rate} Hz. "
            "Resample the input audio before caching features."
        )
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if waveform.size == 0:
        return np.zeros(CROP_SAMPLES, dtype=np.float32)
    if waveform.size < CROP_SAMPLES:
        waveform = np.tile(waveform, CROP_SAMPLES // waveform.size + 1)
    return np.ascontiguousarray(waveform[:CROP_SAMPLES], dtype=np.float32)


def _read_audio(file_obj: Any, display_name: str) -> np.ndarray:
    try:
        waveform, sample_rate = sf.read(file_obj, dtype="float32")
    except Exception as exc:
        raise RuntimeError(f"Could not decode audio {display_name}: {exc}") from exc
    return prepare_waveform(waveform, sample_rate)


class FrozenXLSR:
    def __init__(self, model_name: str, device: torch.device):
        try:
            from transformers import AutoFeatureExtractor, AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "The feature-caching step requires transformers. "
                "Install requirements-zahira.txt in the notebook's Python environment."
            ) from exc
        self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name, output_hidden_states=True
        ).to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = device
        self.model_name = model_name
        self.hidden_size = int(self.model.config.hidden_size)
        self.num_layers = int(self.model.config.num_hidden_layers) + 1

    @torch.inference_mode()
    def encode(self, waveforms: list[np.ndarray]) -> np.ndarray:
        inputs = self.extractor(
            waveforms,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs["input_values"].to(self.device)
        outputs = self.model(input_values, output_hidden_states=True)
        if outputs.hidden_states is None:
            raise RuntimeError("XLS-R did not return hidden states.")
        statistics = []
        for hidden in outputs.hidden_states:
            mean = hidden.mean(dim=1)
            std = hidden.std(dim=1, unbiased=False)
            statistics.append(torch.cat((mean, std), dim=-1))
        return torch.stack(statistics, dim=1).to(
            device="cpu", dtype=torch.float16
        ).numpy()


def _cache_paths(output_dir: Path, split: str) -> tuple[Path, Path, Path, Path]:
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return (
        cache_dir / f"{split}.features.f16",
        cache_dir / f"{split}.json",
        cache_dir / f"{split}.partial.f16",
        cache_dir / f"{split}.partial.json",
    )


def _source_paths(split: str, audio_dir: Path | None, archive_dir: Path) -> list[Path]:
    if audio_dir is not None:
        if not audio_dir.is_dir():
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
        return [audio_dir]
    archive_prefix = {"train": "flac_T_", "dev": "flac_D_", "eval": "flac_E_"}[split]
    archives = sorted(archive_dir.glob(f"{archive_prefix}*.tar"))
    if not archives:
        raise FileNotFoundError(
            f"No {archive_prefix}*.tar audio archives found under {archive_dir}. "
            "Download the complete ASVspoof shards there or pass --audio-dir."
        )
    return archives


def cache_split(
    split: str,
    project_dir: Path,
    audio_dir: Path | None,
    archive_dir: Path | None,
    model_name: str,
    batch_size: int,
) -> None:
    paths = project_paths(project_dir)
    frame = load_protocol(split, paths)
    output_dir = paths["outputs"]
    feature_path, metadata_path, partial_path, partial_metadata_path = _cache_paths(
        output_dir, split
    )
    if feature_path.is_file() and metadata_path.is_file():
        print(f"[skip] {split} feature cache already exists: {feature_path}")
        return
    archive_dir = archive_dir or paths["archives"]
    sources = _source_paths(split, audio_dir, archive_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = FrozenXLSR(model_name, device)
    shape = (len(frame), encoder.num_layers, encoder.hidden_size * 2)
    required_bytes = math.prod(shape) * np.dtype(np.float16).itemsize
    partial_exists = partial_path.is_file() and partial_metadata_path.is_file()
    free_bytes = shutil.disk_usage(feature_path.parent).free
    if not partial_exists and required_bytes > free_bytes:
        raise OSError(
            f"{split} feature cache needs about {required_bytes / 1e9:.1f} GB, "
            f"but only {free_bytes / 1e9:.1f} GB is free."
        )
    filenames = frame["filename"].tolist()
    wanted = {name: index for index, name in enumerate(filenames)}
    if partial_exists:
        progress = json.loads(partial_metadata_path.read_text(encoding="utf-8"))
        if (
            progress["split"] != split
            or progress["model_name"] != model_name
            or progress["shape"] != list(shape)
            or progress["filenames"] != filenames
        ):
            raise ValueError(
                f"Existing partial {split} cache does not match this protocol/model. "
                f"Remove {partial_path} and {partial_metadata_path} to start over."
            )
        completed_sources = set(progress["completed_sources"])
        completed_indices = progress["completed_indices"]
        features = np.memmap(partial_path, mode="r+", dtype=np.float16, shape=shape)
        print(
            f"Resuming {split} cache: {len(completed_indices):,}/{len(frame):,} "
            "files already cached"
        )
    else:
        completed_sources = set()
        completed_indices = []
        features = np.memmap(partial_path, mode="w+", dtype=np.float16, shape=shape)
        progress = {
            "split": split,
            "shape": list(shape),
            "model_name": model_name,
            "filenames": filenames,
            "completed_sources": [],
            "completed_indices": [],
        }
        partial_metadata_path.write_text(
            json.dumps(progress, separators=(",", ":")), encoding="utf-8"
        )
    seen = np.zeros(len(frame), dtype=bool)
    seen[completed_indices] = True
    pending_indices: list[int] = []
    pending_waveforms: list[np.ndarray] = []
    total_written = 0

    def flush_batch() -> None:
        nonlocal total_written
        if not pending_indices:
            return
        encoded = encoder.encode(pending_waveforms)
        if encoded.shape[1:] != shape[1:]:
            raise RuntimeError(
                f"Unexpected feature shape {encoded.shape[1:]}; expected {shape[1:]}."
            )
        features[pending_indices] = encoded
        seen[pending_indices] = True
        total_written += len(pending_indices)
        pending_indices.clear()
        pending_waveforms.clear()

    try:
        for source in sources:
            source_id = str(source.resolve())
            if source_id in completed_sources:
                print(f"[skip] already cached {source.name}")
                continue
            source_indices: set[int] = set()
            if source.is_dir():
                audio_files = sorted(
                    path
                    for path in source.rglob("*")
                    if path.is_file() and path.suffix.lower() in {".flac", ".wav"}
                )
                for audio_path in audio_files:
                    index = wanted.get(audio_path.stem)
                    if index is None:
                        continue
                    if seen[index] or index in source_indices:
                        raise ValueError(f"Duplicate audio file for {audio_path.stem}")
                    with audio_path.open("rb") as audio_file:
                        pending_waveforms.append(_read_audio(audio_file, str(audio_path)))
                    pending_indices.append(index)
                    source_indices.add(index)
                    if len(pending_indices) >= batch_size:
                        flush_batch()
            else:
                try:
                    with tarfile.open(source, mode="r:*") as archive:
                        for member in archive:
                            if not member.isfile():
                                continue
                            filename = Path(member.name).stem
                            index = wanted.get(filename)
                            if index is None:
                                continue
                            if seen[index] or index in source_indices:
                                raise ValueError(f"Duplicate audio file for {filename}")
                            audio_file = archive.extractfile(member)
                            if audio_file is None:
                                raise RuntimeError(f"Could not read {member.name} from {source}")
                            with audio_file:
                                pending_waveforms.append(
                                    _read_audio(audio_file, f"{source}:{member.name}")
                                )
                            pending_indices.append(index)
                            source_indices.add(index)
                            if len(pending_indices) >= batch_size:
                                flush_batch()
                except (tarfile.TarError, EOFError) as exc:
                    raise RuntimeError(
                        f"Could not read complete archive {source}. "
                        "A truncated shard must be downloaded again."
                    ) from exc
            flush_batch()
            completed_indices.extend(sorted(source_indices))
            completed_sources.add(source_id)
            features.flush()
            progress["completed_sources"] = sorted(completed_sources)
            progress["completed_indices"] = completed_indices
            temporary_progress = partial_metadata_path.with_suffix(".json.tmp")
            temporary_progress.write_text(
                json.dumps(progress, separators=(",", ":")), encoding="utf-8"
            )
            os.replace(temporary_progress, partial_metadata_path)
            print(f"Processed {source.name}: {int(seen.sum()):,}/{len(frame):,} files")

        missing = [filenames[index] for index in np.flatnonzero(~seen)]
        if missing:
            features.flush()
            del features
            print(
                f"Cached {int(seen.sum()):,}/{len(frame):,} {split} files. "
                f"Missing {len(missing):,}; first missing IDs: {missing[:10]}. "
                "Add the remaining complete shard(s) and rerun this cache command."
            )
            return
        features.flush()
        del features
        os.replace(partial_path, feature_path)
        metadata = {
            "split": split,
            "shape": list(shape),
            "dtype": "float16",
            "model_name": model_name,
            "filenames": filenames,
            "speakers": frame["SPEAKER_ID"].tolist(),
            "keys": frame["KEY"].tolist(),
        }
        temporary_metadata = metadata_path.with_suffix(".json.tmp")
        temporary_metadata.write_text(
            json.dumps(metadata, separators=(",", ":")), encoding="utf-8"
        )
        os.replace(temporary_metadata, metadata_path)
        partial_metadata_path.unlink()
        print(f"Saved {split} features to {feature_path}")
    except Exception:
        if "features" in locals():
            features.flush()
            del features
        raise


def _load_cache(output_dir: Path, split: str) -> tuple[np.memmap, dict[str, Any]]:
    feature_path, metadata_path, _, _ = _cache_paths(output_dir, split)
    if not feature_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing {split} cache. Run: python zahira_ssl.py cache --split {split}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    features = np.memmap(
        feature_path,
        mode="r",
        dtype=np.dtype(metadata["dtype"]),
        shape=tuple(metadata["shape"]),
    )
    return features, metadata


class FeatureDataset(Dataset):
    def __init__(self, features: np.memmap, keys: list[str]):
        self.features = features
        self.labels = np.asarray([1 if key == "bonafide" else 0 for key in keys], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        values = np.asarray(self.features[index], dtype=np.float32)
        return torch.from_numpy(values), torch.tensor(self.labels[index])


class LayerWeightedHead(nn.Module):
    def __init__(self, num_layers: int, feature_dim: int):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.layer_weights, dim=0)
        pooled = (features * weights[None, :, None]).sum(dim=1)
        return self.classifier(pooled)


def _score_frame(
    logits: np.ndarray, filenames: list[str]
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "filename": filenames,
            "cm-score": logits[:, 1] - logits[:, 0],
            "logit_spoof": logits[:, 0],
            "logit_bonafide": logits[:, 1],
        }
    )


def _save_scores(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    frame.to_csv(temporary_path, sep="\t", index=False)
    os.replace(temporary_path, output_path)


def _predict(
    model: LayerWeightedHead,
    features: np.memmap,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    dataset = FeatureDataset(features, ["spoof"] * len(features))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    batches = []
    with torch.inference_mode():
        for inputs, _ in loader:
            batches.append(model(inputs.to(device)).cpu().numpy())
    return np.concatenate(batches, axis=0)


def train_heads(
    project_dir: Path,
    seeds: tuple[int, ...],
    epochs: int,
    batch_size: int,
) -> None:
    paths = project_paths(project_dir)
    train_features, train_meta = _load_cache(paths["outputs"], "train")
    dev_features, dev_meta = _load_cache(paths["outputs"], "dev")
    if train_meta["model_name"] != dev_meta["model_name"]:
        raise ValueError("Train and dev caches were made with different XLS-R models.")
    if train_meta["shape"][1:] != dev_meta["shape"][1:]:
        raise ValueError("Train and dev feature shapes do not match.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = paths["outputs"] / "models"
    score_dir = paths["outputs"] / "scores"
    model_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)
    dev_logits_by_seed = []

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = LayerWeightedHead(
            num_layers=train_features.shape[1],
            feature_dim=train_features.shape[2],
        ).to(device)
        train_data = FeatureDataset(train_features, train_meta["keys"])
        dev_data = FeatureDataset(dev_features, dev_meta["keys"])
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        dev_loader = DataLoader(
            dev_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best_loss = float("inf")
        best_state = None
        best_epoch = 0
        stale_epochs = 0
        history = []

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss_sum = 0.0
            train_count = 0
            for inputs, targets in train_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(inputs), targets)
                loss.backward()
                optimizer.step()
                train_loss_sum += float(loss.item()) * len(targets)
                train_count += len(targets)

            model.eval()
            dev_loss_sum = 0.0
            dev_count = 0
            with torch.inference_mode():
                for inputs, targets in dev_loader:
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    loss = nn.functional.cross_entropy(model(inputs), targets)
                    dev_loss_sum += float(loss.item()) * len(targets)
                    dev_count += len(targets)
            train_loss = train_loss_sum / train_count
            dev_loss = dev_loss_sum / dev_count
            history.append(
                {"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss}
            )
            print(
                f"seed={seed} epoch={epoch}/{epochs} "
                f"train_loss={train_loss:.5f} dev_loss={dev_loss:.5f}"
            )
            if dev_loss < best_loss:
                best_loss = dev_loss
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= 3:
                    break

        if best_state is None:
            raise RuntimeError(f"Training seed {seed} did not produce a checkpoint.")
        model.load_state_dict(best_state)
        checkpoint_path = model_dir / f"xlsr_seed_{seed}.pt"
        torch.save(
            {
                "seed": seed,
                "model_name": train_meta["model_name"],
                "num_layers": train_features.shape[1],
                "feature_dim": train_features.shape[2],
                "best_epoch": best_epoch,
                "best_dev_loss": best_loss,
                "state_dict": best_state,
            },
            checkpoint_path,
        )
        logits = _predict(model, dev_features, device, batch_size)
        _save_scores(
            _score_frame(logits, dev_meta["filenames"]),
            score_dir / f"xlsr_dev_seed_{seed}.tsv",
        )
        dev_logits_by_seed.append(logits)
        (model_dir / f"xlsr_seed_{seed}_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(f"Saved seed {seed} checkpoint: {checkpoint_path}")

    ensemble_logits = np.mean(dev_logits_by_seed, axis=0)
    _save_scores(_score_frame(ensemble_logits, dev_meta["filenames"]), score_dir / "xlsr_dev.tsv")
    print(f"Saved ensemble dev scores: {score_dir / 'xlsr_dev.tsv'}")


def score_split(
    split: str,
    project_dir: Path,
    seeds: tuple[int, ...] | None,
    batch_size: int,
) -> None:
    paths = project_paths(project_dir)
    features, metadata = _load_cache(paths["outputs"], split)
    model_dir = paths["outputs"] / "models"
    score_dir = paths["outputs"] / "scores"
    checkpoint_paths = (
        [model_dir / f"xlsr_seed_{seed}.pt" for seed in seeds]
        if seeds
        else sorted(model_dir.glob("xlsr_seed_*.pt"))
    )
    checkpoint_paths = [
        path for path in checkpoint_paths if not path.name.endswith("_history.json")
    ]
    if not checkpoint_paths:
        raise FileNotFoundError(f"No trained checkpoints found in {model_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_logits = []
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["model_name"] != metadata["model_name"]:
            raise ValueError(f"{checkpoint_path} was trained with a different XLS-R model.")
        model = LayerWeightedHead(
            checkpoint["num_layers"], checkpoint["feature_dim"]
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        logits = _predict(model, features, device, batch_size)
        seed = int(checkpoint["seed"])
        _save_scores(
            _score_frame(logits, metadata["filenames"]),
            score_dir / f"xlsr_{split}_seed_{seed}.tsv",
        )
        all_logits.append(logits)
        print(f"Scored {split} with seed {seed}")
    ensemble_logits = np.mean(all_logits, axis=0)
    _save_scores(
        _score_frame(ensemble_logits, metadata["filenames"]),
        score_dir / f"xlsr_{split}.tsv",
    )
    print(f"Saved ensemble {split} scores: {score_dir / f'xlsr_{split}.tsv'}")


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    scores = torch.tensor(logits, dtype=torch.float64)
    labels = torch.tensor(targets, dtype=torch.long)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(1e-3, 1e3)
        loss = nn.functional.cross_entropy(scores / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(1e-3, 1e3).item())


def expected_calibration_error(
    probabilities: np.ndarray, targets: np.ndarray, bins: int = 15
) -> tuple[float, pd.DataFrame]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    ece = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        selected = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        count = int(selected.sum())
        if count == 0:
            rows.append(
                {"bin_lower": lower, "bin_upper": upper, "count": 0,
                 "confidence": np.nan, "accuracy": np.nan}
            )
            continue
        confidence = float(probabilities[selected].mean())
        accuracy = float(targets[selected].mean())
        ece += count / len(targets) * abs(confidence - accuracy)
        rows.append(
            {"bin_lower": lower, "bin_upper": upper, "count": count,
             "confidence": confidence, "accuracy": accuracy}
        )
    return ece, pd.DataFrame(rows)


def selective_metrics(
    probabilities_bonafide: np.ndarray, targets: np.ndarray
) -> tuple[dict[str, float], pd.DataFrame]:
    predictions = (probabilities_bonafide >= 0.5).astype(np.int64)
    errors = predictions != targets
    confidence = np.maximum(probabilities_bonafide, 1.0 - probabilities_bonafide)
    order = np.argsort(-confidence, kind="stable")
    sorted_errors = errors[order].astype(np.float64)
    count = np.arange(1, len(targets) + 1)
    coverage = count / len(targets)
    risk = np.cumsum(sorted_errors) / count
    oracle_errors = np.sort(errors.astype(np.float64))
    oracle_risk = np.cumsum(oracle_errors) / count
    aurc = float((risk.sum() - risk[-1] / 2.0) / len(targets))
    oracle_aurc = float((oracle_risk.sum() - oracle_risk[-1] / 2.0) / len(targets))

    report: dict[str, float] = {
        "aurc": aurc,
        "oracle_aurc": oracle_aurc,
        "excess_aurc": aurc - oracle_aurc,
    }
    for target_coverage in (0.8, 0.9):
        retained = int(math.ceil(target_coverage * len(targets)))
        report[f"coverage_at_{int(target_coverage * 100)}"] = retained / len(targets)
        report[f"risk_at_{int(target_coverage * 100)}"] = float(
            sorted_errors[:retained].mean()
        )
    curve = pd.DataFrame(
        {
            "coverage": coverage,
            "risk": risk,
            "confidence": confidence[order],
            "error": sorted_errors.astype(np.int64),
        }
    )
    return report, curve


def _official_metrics(scores: np.ndarray, keys: np.ndarray) -> dict[str, float]:
    evaluation_dir = (
        Path(__file__).resolve().parent
        / "local_data"
        / "asvspoof5"
        / "evaluation-package"
    )
    if not evaluation_dir.is_dir():
        raise FileNotFoundError(f"Official ASVspoof evaluation package missing: {evaluation_dir}")
    import sys

    sys.path.insert(0, str(evaluation_dir))
    try:
        from calculate_metrics import calculate_minDCF_EER_CLLR_actDCF
    except ImportError as exc:
        raise RuntimeError(
            "Could not import the official ASVspoof metric package. "
            "Install its dependencies in the active Python environment."
        ) from exc
    min_dcf, eer, cllr, actual_dcf = calculate_minDCF_EER_CLLR_actDCF(
        scores, keys, output_file=os.devnull, printout=False
    )
    return {
        "minDCF": float(min_dcf),
        "EER_percent": float(eer * 100.0),
        "CLLR": float(cllr),
        "actDCF": float(actual_dcf),
    }


def _plot_calibration(
    bins: pd.DataFrame, curve: pd.DataFrame, output_dir: Path, split: str
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Saving calibration plots requires matplotlib. Install requirements-zahira.txt."
        ) from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    populated = bins[bins["count"] > 0]
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    axis.plot(populated["confidence"], populated["accuracy"], marker="o", label="XLS-R")
    axis.set(xlabel="Mean predicted bona fide probability", ylabel="Bona fide fraction",
             title=f"{split.upper()} reliability diagram", xlim=(0, 1), ylim=(0, 1))
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"xlsr_{split}_reliability.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(curve["coverage"], curve["risk"], label="Confidence-ranked")
    axis.set(xlabel="Coverage", ylabel="Classification error among retained files",
             title=f"{split.upper()} risk–coverage", xlim=(0, 1), ylim=(0, 1))
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"xlsr_{split}_risk_coverage.png", dpi=160)
    plt.close(fig)


def calibrate_split(split: str, project_dir: Path) -> None:
    paths = project_paths(project_dir)
    score_dir = paths["outputs"] / "scores"
    dev_path = score_dir / "xlsr_dev.tsv"
    split_path = score_dir / f"xlsr_{split}.tsv"
    if not dev_path.is_file():
        raise FileNotFoundError(f"Dev scores are missing: {dev_path}. Train the head first.")
    if not split_path.is_file():
        raise FileNotFoundError(f"Raw {split} scores are missing: {split_path}. Score that split first.")
    dev_features, dev_meta = _load_cache(paths["outputs"], "dev")
    del dev_features
    target_features, target_meta = _load_cache(paths["outputs"], split)
    del target_features
    dev_scores = pd.read_csv(dev_path, sep="\t")
    target_scores = pd.read_csv(split_path, sep="\t")
    dev = pd.DataFrame(
        {"filename": dev_meta["filenames"], "KEY": dev_meta["keys"]}
    ).merge(dev_scores, on="filename", validate="one_to_one")
    target = pd.DataFrame(
        {"filename": target_meta["filenames"], "KEY": target_meta["keys"]}
    ).merge(target_scores, on="filename", validate="one_to_one")
    if len(dev) != len(dev_meta["filenames"]) or len(target) != len(target_meta["filenames"]):
        raise ValueError("Scores did not align with the protocol filenames.")
    dev_logits = dev[["logit_spoof", "logit_bonafide"]].to_numpy(dtype=np.float64)
    dev_targets = (dev["KEY"].to_numpy() == "bonafide").astype(np.int64)
    temperature = fit_temperature(dev_logits, dev_targets)
    logits = target[["logit_spoof", "logit_bonafide"]].to_numpy(dtype=np.float64)
    targets = (target["KEY"].to_numpy() == "bonafide").astype(np.int64)
    raw_probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    calibrated_logits = logits / temperature
    probabilities = torch.softmax(torch.tensor(calibrated_logits), dim=1).numpy()[:, 1]
    raw_ece, _ = expected_calibration_error(raw_probabilities, targets)
    ece, bins = expected_calibration_error(probabilities, targets)
    selective, curve = selective_metrics(probabilities, targets)
    raw_metric = _official_metrics(logits[:, 1] - logits[:, 0], target["KEY"].to_numpy())
    calibrated_metric = _official_metrics(
        calibrated_logits[:, 1] - calibrated_logits[:, 0], target["KEY"].to_numpy()
    )
    metrics = {
        "split": split,
        "temperature_fitted_on": "dev",
        "temperature": temperature,
        "raw": {**raw_metric, "ECE": raw_ece},
        "calibrated": {**calibrated_metric, "ECE": ece},
        "selective_prediction": selective,
    }
    output_dir = paths["outputs"]
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics" / f"xlsr_{split}_calibration.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    bins.to_csv(output_dir / "metrics" / f"xlsr_{split}_reliability.tsv",
                sep="\t", index=False)
    curve.to_csv(output_dir / "metrics" / f"xlsr_{split}_risk_coverage.tsv",
                 sep="\t", index=False)
    calibrated_frame = target[["filename"]].copy()
    calibrated_frame["cm-score"] = calibrated_logits[:, 1] - calibrated_logits[:, 0]
    calibrated_frame["logit_spoof"] = calibrated_logits[:, 0]
    calibrated_frame["logit_bonafide"] = calibrated_logits[:, 1]
    calibrated_frame["bonafide_probability"] = probabilities
    _save_scores(calibrated_frame, score_dir / f"xlsr_{split}_calibrated.tsv")
    _plot_calibration(bins, curve, output_dir / "plots", split)
    print(json.dumps(metrics, indent=2))


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(part) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Seeds must be comma-separated integers.") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required.")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    project_default = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=project_default)
    commands = parser.add_subparsers(dest="command", required=True)

    cache_parser = commands.add_parser("cache", help="Extract and cache frozen XLS-R features.")
    cache_parser.add_argument("--split", choices=("train", "dev", "eval"), required=True)
    cache_parser.add_argument("--audio-dir", type=Path)
    cache_parser.add_argument("--archive-dir", type=Path)
    cache_parser.add_argument("--model-name", default=DEFAULT_MODEL)
    cache_parser.add_argument("--batch-size", type=int, default=2)

    train_parser = commands.add_parser("train", help="Train the 3-seed layer-weighted heads.")
    train_parser.add_argument("--seeds", type=_parse_seeds, default=DEFAULT_SEEDS)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=256)

    score_parser = commands.add_parser("score", help="Score a cached split with trained heads.")
    score_parser.add_argument("--split", choices=("dev", "eval"), required=True)
    score_parser.add_argument("--seeds", type=_parse_seeds)
    score_parser.add_argument("--batch-size", type=int, default=256)

    calibrate_parser = commands.add_parser(
        "calibrate", help="Apply dev-fitted temperature scaling and report calibration/abstention."
    )
    calibrate_parser.add_argument("--split", choices=("dev", "eval"), required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "cache":
        if args.batch_size < 1:
            raise SystemExit("--batch-size must be positive.")
        cache_split(
            args.split,
            args.project_dir,
            args.audio_dir,
            args.archive_dir,
            args.model_name,
            args.batch_size,
        )
    elif args.command == "train":
        if args.epochs < 1 or args.batch_size < 1:
            raise SystemExit("--epochs and --batch-size must be positive.")
        train_heads(args.project_dir, args.seeds, args.epochs, args.batch_size)
    elif args.command == "score":
        if args.batch_size < 1:
            raise SystemExit("--batch-size must be positive.")
        score_split(args.split, args.project_dir, args.seeds, args.batch_size)
    elif args.command == "calibrate":
        calibrate_split(args.split, args.project_dir)


if __name__ == "__main__":
    main()
