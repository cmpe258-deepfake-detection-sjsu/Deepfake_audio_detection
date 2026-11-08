import json
import tempfile
import tarfile
from pathlib import Path
from unittest.mock import patch
import unittest

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from zahira_ssl import (
    CROP_SAMPLES,
    LayerWeightedHead,
    _load_cache,
    cache_split,
    expected_calibration_error,
    fit_temperature,
    prepare_waveform,
    project_paths,
    score_split,
    selective_metrics,
    train_heads,
)


class ZahiraPipelineTests(unittest.TestCase):
    def test_prepare_waveform_repeats_short_clip_and_truncates_long_clip(self):
        short = prepare_waveform(np.array([0.25, -0.5], dtype=np.float32), 16_000)
        self.assertEqual(short.shape, (CROP_SAMPLES,))
        np.testing.assert_array_equal(short[:4], [0.25, -0.5, 0.25, -0.5])

        long = prepare_waveform(np.ones(CROP_SAMPLES + 4, dtype=np.float32), 16_000)
        self.assertEqual(long.shape, (CROP_SAMPLES,))

    def test_prepare_waveform_rejects_unexpected_sample_rate(self):
        with self.assertRaisesRegex(ValueError, "Expected 16000 Hz"):
            prepare_waveform(np.zeros(10, dtype=np.float32), 8_000)

    def test_layer_weighted_head_returns_spoof_and_bonafide_logits(self):
        model = LayerWeightedHead(num_layers=4, feature_dim=8)
        logits = model(torch.zeros(3, 4, 8))
        self.assertEqual(tuple(logits.shape), (3, 2))
        weight_sum = torch.softmax(model.layer_weights, dim=0).sum().detach().item()
        self.assertAlmostEqual(weight_sum, 1.0)

    def test_temperature_scaling_reduces_overconfidence(self):
        logits = np.array([[0.0, 4.0], [4.0, 0.0], [0.0, 3.0], [3.0, 0.0]])
        targets = np.array([1, 0, 0, 1])
        temperature = fit_temperature(logits, targets)
        self.assertGreater(temperature, 1.0)
        self.assertTrue(np.isfinite(temperature))

    def test_calibration_and_selective_metrics(self):
        probabilities = np.array([0.9, 0.8, 0.4, 0.1])
        targets = np.array([1, 0, 1, 0])
        ece, bins = expected_calibration_error(probabilities, targets, bins=5)
        self.assertGreaterEqual(ece, 0.0)
        self.assertAlmostEqual(float(bins["count"].sum()), len(targets))

        report, curve = selective_metrics(probabilities, targets)
        self.assertEqual(len(curve), len(targets))
        self.assertGreaterEqual(report["aurc"], report["oracle_aurc"])
        self.assertEqual(report["coverage_at_80"], 1.0)
        self.assertEqual(report["coverage_at_90"], 1.0)

    def test_feature_cache_reads_archive_audio_and_writes_complete_cache(self):
        class FakeEncoder:
            def __init__(self, model_name, device):
                self.model_name = model_name
                self.hidden_size = 3
                self.num_layers = 2

            def encode(self, waveforms):
                return np.ones((len(waveforms), self.num_layers, 6), dtype=np.float16)

        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            paths = project_paths(project)
            paths["data"].mkdir(parents=True)
            paths["data"].joinpath("ASVspoof5.dev.track_1.tsv").write_text(
                "D_0001 D_0000000001 F - - - - A09 bonafide -\n"
                "D_0002 D_0000000002 M - - - - A10 spoof -\n",
                encoding="utf-8",
            )
            archive_dir = project / "archives"
            archive_dir.mkdir()

            def write_archive(shard, audio_id):
                audio_path = project / f"{audio_id}.flac"
                sf.write(audio_path, np.zeros(200, dtype=np.float32), 16_000, format="FLAC")
                with tarfile.open(archive_dir / shard, "w") as archive:
                    archive.add(audio_path, arcname=f"nested/{audio_id}.flac")

            with patch("zahira_ssl.FrozenXLSR", FakeEncoder):
                write_archive("flac_D_aa.tar", "D_0000000001")
                cache_split(
                    "dev",
                    project,
                    audio_dir=None,
                    archive_dir=archive_dir,
                    model_name="test-model",
                    batch_size=1,
                )
                self.assertFalse((paths["outputs"] / "cache" / "dev.json").exists())
                write_archive("flac_D_ab.tar", "D_0000000002")
                cache_split(
                    "dev",
                    project,
                    audio_dir=None,
                    archive_dir=archive_dir,
                    model_name="test-model",
                    batch_size=1,
                )
            features, metadata = _load_cache(paths["outputs"], "dev")
            self.assertEqual(tuple(features.shape), (2, 2, 6))
            np.testing.assert_array_equal(features[:], np.ones((2, 2, 6), dtype=np.float16))
            self.assertEqual(
                metadata["filenames"], ["D_0000000001", "D_0000000002"]
            )
            del features

    def test_training_saves_three_seed_format_scores_and_scoring_loads_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            paths = project_paths(project)
            cache_dir = paths["outputs"] / "cache"
            cache_dir.mkdir(parents=True)

            def write_cache(split, keys):
                features = np.memmap(
                    cache_dir / f"{split}.features.f16",
                    mode="w+",
                    dtype=np.float16,
                    shape=(len(keys), 2, 4),
                )
                features[:] = np.random.default_rng(42).normal(
                    size=(len(keys), 2, 4)
                ).astype(np.float16)
                features.flush()
                del features
                metadata = {
                    "split": split,
                    "shape": [len(keys), 2, 4],
                    "dtype": "float16",
                    "model_name": "test-model",
                    "filenames": [f"{split}_{index}" for index in range(len(keys))],
                    "speakers": ["S1"] * len(keys),
                    "keys": keys,
                }
                (cache_dir / f"{split}.json").write_text(
                    json.dumps(metadata), encoding="utf-8"
                )

            write_cache("train", ["bonafide", "spoof", "bonafide", "spoof"])
            write_cache("dev", ["bonafide", "spoof", "bonafide", "spoof"])
            train_heads(project, seeds=(17,), epochs=1, batch_size=2)
            write_cache("eval", ["bonafide", "spoof"])
            score_split("eval", project, seeds=(17,), batch_size=2)

            output = pd.read_csv(
                paths["outputs"] / "scores" / "xlsr_eval.tsv", sep="\t"
            )
            self.assertEqual(output["filename"].tolist(), ["eval_0", "eval_1"])
            self.assertIn("cm-score", output.columns)
            self.assertEqual(len(output), 2)


if __name__ == "__main__":
    unittest.main()
