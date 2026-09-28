"""CPU checks that do not download model weights."""

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from saga.cli import parse_args
from saga.data import load_pairs
from saga.hotspots import build_schedule
from saga.optimization import optimize
from saga.text import filter_generated_tokens


def test_supplied_main_configuration():
    args = parse_args(
        [
            "--dataset",
            "aadcd",
            "--method",
            "ours",
            "--seed",
            "43",
            "--vlm_attention_extractor",
            "qwen3_vl",
            "--layer_idx",
            "29",
            "--no-use_smart_resize",
            "--use_content_words",
            "--one_to_one",
            "--progressive_mode",
            "--hotspot_prob",
            "0.5",
        ]
    )
    assert (args.attack_steps, args.attack_epsilon, args.optimization_seed) == (300, 16.0, 2023)


def test_baselines_not_exposed():
    with pytest.raises(SystemExit):
        parse_args(["--method", "m_attack"])


def test_packaged_data_are_complete():
    root = Path(__file__).resolve().parents[1]
    data = root / "input/images/aadcd"
    if len(list(data.glob("*.png"))) != 1000:
        pytest.skip("Clean source images not installed; server ZIP includes them")
    pairs = load_pairs(data, root / "input/coco.jsonl", root / "input/coco.txt")
    assert len(pairs) == 1000
    assert len({path.stem for path, _ in pairs}) == 1000
    assert [path.stem for path, _ in pairs][:4] == ["0", "1", "10", "100"]


def test_missing_caption_is_an_error(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (4, 4)).save(images / "42.png")
    captions = tmp_path / "captions.jsonl"
    captions.write_text(json.dumps({"file_name": "1.jpg", "caption": "wrong target"}) + "\n")
    with pytest.raises(ValueError, match="Missing JSONL"):
        load_pairs(images, captions, num_images=1)


def test_duplicate_caption_is_an_error(tmp_path):
    captions = tmp_path / "captions.jsonl"
    captions.write_text((json.dumps({"file_name": "1.jpg", "caption": "target"}) + "\n") * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        load_pairs(tmp_path, captions, num_images=1)


def test_schedule_expands_to_full_image():
    schedule = build_schedule(np.random.default_rng(43).random((14, 14)).astype(np.float32))
    assert len(schedule) == 28  # 9*3 rectangles and one full-image rectangle.
    assert schedule[-1]["box"] == (0.0, 0.0, 1.0, 1.0)
    assert [s["threshold"] for s in schedule] == sorted(s["threshold"] for s in schedule)
    assert all(
        0 <= t < 1 and 0 <= left < 1 and 0 < h <= 1 - t and 0 < w <= 1 - left
        for t, left, h, w in [s["box"] for s in schedule]
    )


def test_invalid_attention_rejected():
    with pytest.raises(ValueError, match="finite"):
        build_schedule(np.array([[np.nan]]))


class ToyExtractor:
    def __call__(self, image, kind):
        return image.mean()


class ToyObjective:
    def set_ground_truth(self, target, kind):
        pass

    def __call__(self, features):
        return features


def test_updates_stay_in_hotspot_and_within_budget():
    torch.manual_seed(9)
    random.seed(9)
    image = torch.full((1, 3, 224, 224), 100.0)
    schedule = [{"box": (0.0, 0.0, 0.5, 0.5), "threshold": 0.25}]
    clean, adv, losses = optimize(
        image, "target", schedule, ToyExtractor(), ToyObjective(), device="cpu", steps=20, epsilon=4.0
    )
    delta = (adv - clean) * 255
    assert delta.max() <= 4.0001
    assert delta.max() > 0
    assert torch.count_nonzero(delta[:, :, 112:, :]) == 0
    assert torch.count_nonzero(delta[:, :, :, 112:]) == 0
    assert len(losses) == 20
    assert adv.min() >= 0 and adv.max() <= 1


def test_zero_epsilon_keeps_clean_pixels():
    image = torch.full((1, 3, 224, 224), 100.0)
    clean, adv, _ = optimize(
        image,
        "target",
        [{"box": (0.0, 0.0, 1.0, 1.0)}],
        ToyExtractor(),
        ToyObjective(),
        device="cpu",
        steps=1,
        epsilon=0.0,
    )
    assert torch.equal(clean, adv)


def test_content_words_and_empty_filter_fallback():
    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: ["the", "cat", "and"][int(ids[0])])
    assert filter_generated_tokens(tokenizer, [0, 1, 2], True) == [1]
    assert filter_generated_tokens(tokenizer, [0, 2], True) == [0, 1]
