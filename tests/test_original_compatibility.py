"""Regression cases for original-main sampling, tensor layout and model placement."""

import json
import random
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image

from saga import attention, pipeline
from saga.data import load_pairs


def test_subset_samples_before_sorting(tmp_path, monkeypatch):
    names = ["729.png", "133.png", "31.png", "2.png", "10.png"]
    for name in names:
        (tmp_path / name).touch()
    targets = tmp_path / "targets.jsonl"
    targets.write_text("\n".join(json.dumps({"file_name": n, "caption": n}) for n in names))
    monkeypatch.setattr("saga.data.os.listdir", lambda _: names + ["ignored.txt"])
    expected = sorted(random.Random(43).sample(names, 3), key=lambda n: n.rsplit(".", 1)[0])
    assert [path.name for path, _ in load_pairs(tmp_path, targets, num_images=3)] == expected


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rgb_preprocessing_preserves_original_nchw_layout(dtype):
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        pixels = np.arange(7 * 11 * 3, dtype=np.uint8).reshape(7, 11, 3)
        result = pipeline.preprocess_image(Image.fromarray(pixels))
        expected = torch.tensor(pixels.transpose(2, 0, 1).copy(), dtype=dtype).unsqueeze(0)
        assert torch.equal(result, expected)
        assert result.stride() == (231, 77, 11, 1)
        assert result.is_contiguous()
        assert result.dtype == dtype
    finally:
        torch.set_default_dtype(original_dtype)


def test_distinct_original_seed_helpers(monkeypatch):
    calls = []
    for name in ["manual_seed", "manual_seed_all"]:
        monkeypatch.setattr(torch.cuda, name, lambda seed, name=name: calls.append((name, seed)))
    # Isolate explicit CUDA calls from torch.manual_seed's own implementation.
    monkeypatch.setattr(torch, "manual_seed", lambda seed: calls.append(("torch", seed)))
    monkeypatch.setenv("PYTHONHASHSEED", "prior")
    pipeline.set_seed(43)
    assert calls == [("torch", 43), ("manual_seed", 43), ("manual_seed_all", 43)]
    calls.clear()
    pipeline.set_environment(2023)
    assert calls == [("torch", 2023), ("manual_seed", 2023)]
    import os

    assert os.environ["PYTHONHASHSEED"] == "2023"
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark


def test_attention_uses_auto_placement_and_model_input_device(monkeypatch):
    tensor = Mock()
    tensor.to.return_value = tensor
    processor = Mock()
    processor.image_processor = SimpleNamespace(min_pixels=None, max_pixels=None)
    processor.return_value = {"input_ids": tensor}
    model = Mock(device="cpu")
    model.eval.return_value = model
    load = Mock(return_value=model)
    monkeypatch.setattr(attention.AutoProcessor, "from_pretrained", lambda _: processor)
    monkeypatch.setattr(attention.AutoModelForImageTextToText, "from_pretrained", load)
    monkeypatch.setattr(attention, "process_vision_info", lambda *a, **k: ([], None, {}))
    monkeypatch.setattr(
        attention,
        "get_attention_map_from_generated_text_vlm",
        lambda *a, **k: (torch.tensor([[0.1, 0.7], [0.4, 0.2]]), "caption"),
    )
    result, caption = attention.extract_attention(Image.new("RGB", (32, 32)), "cuda:7")
    assert load.call_args.kwargs["device_map"] == "auto"
    tensor.to.assert_called_once_with("cpu")
    assert processor.image_processor.min_pixels == 200704
    assert processor.image_processor.max_pixels == 1003520
    assert caption == "caption"
    assert result.min() == 0 and result.max() == 1
