"""The public attack output contains only images and resumable metadata."""

import json

import numpy as np
import pytest
from PIL import Image

from saga import pipeline
from saga.cli import parse_args


@pytest.fixture
def tiny_run(tmp_path, monkeypatch):
    image = tmp_path / "0.png"
    Image.new("RGB", (224, 224), (100, 100, 100)).save(image)
    args = parse_args(["--save_path", str(tmp_path / "results"), "--note", "test"])
    monkeypatch.setattr(pipeline.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(pipeline.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(pipeline.torch.cuda, "get_device_name", lambda _: "mock GPU")
    monkeypatch.setattr(pipeline, "load_surrogates", lambda _: (None, None))
    monkeypatch.setattr(pipeline, "extract_attention", lambda *a: (np.ones((14, 14)), "clean caption"))
    monkeypatch.setattr(pipeline, "optimize", lambda image, *a, **kw: (image / 255, image / 255, [0.5]))
    return args, [(image, "target")]


def test_only_three_files_and_verified_resume(tiny_run, monkeypatch):
    args, pairs = tiny_run
    pipeline.run(args, pairs)
    output = args.save_path / args.note
    assert sorted(str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()) == [
        "0/adversarial.png",
        "0/metadata.json",
        "0/original.png",
    ]
    metadata = json.loads((output / "0/metadata.json").read_text())
    assert metadata["completed"] and metadata["target"] == "target"
    assert set(metadata["artifacts_sha256"]) == {"original.png", "adversarial.png"}
    args.resume = True
    monkeypatch.setattr(
        pipeline, "extract_attention", lambda *a: pytest.fail("Completed sample was regenerated")
    )
    monkeypatch.setattr(
        pipeline, "load_surrogates", lambda *a: pytest.fail("Completed run loaded model weights")
    )
    pipeline.run(args, pairs)
    args.attack_alpha = 2
    with pytest.raises(ValueError, match="different inputs/settings"):
        pipeline.run(args, pairs)


def test_corrupted_output_is_not_silently_skipped(tiny_run):
    args, pairs = tiny_run
    pipeline.run(args, pairs)
    (args.save_path / args.note / "0/adversarial.png").write_bytes(b"corrupted")
    args.resume = True
    with pytest.raises(RuntimeError, match="changed files"):
        pipeline.run(args, pairs)


def test_interrupted_sample_can_resume(tiny_run, monkeypatch):
    args, pairs = tiny_run
    optimizer = pipeline.optimize

    def interrupted(*a, **kw):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(pipeline, "optimize", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        pipeline.run(args, pairs)
    metadata = args.save_path / args.note / "0/metadata.json"
    assert not json.loads(metadata.read_text())["completed"]
    args.resume = True
    monkeypatch.setattr(pipeline, "optimize", optimizer)
    pipeline.run(args, pairs)
    assert json.loads(metadata.read_text())["completed"]


@pytest.mark.parametrize("state", ["canonical", "initial_partial", "initial_complete", "empty"])
def test_resume_recovers_interrupted_atomic_metadata_write(tiny_run, state):
    args, pairs = tiny_run
    pipeline.run(args, pairs)
    sample = args.save_path / args.note / "0"
    metadata_path = sample / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["completed"] = False
    for name in ["original.png", "adversarial.png"]:
        (sample / name).unlink()
    metadata_path.write_text(json.dumps(metadata))
    temporary = sample / "metadata.json.tmp"
    if state != "empty":
        temporary.write_text(json.dumps(metadata) if state == "initial_complete" else '{"run_id":')
    if state != "canonical":
        metadata_path.unlink()
    args.resume = True
    pipeline.run(args, pairs)
    assert json.loads(metadata_path.read_text())["completed"]
    assert {p.name for p in sample.iterdir()} == {"original.png", "adversarial.png", "metadata.json"}


def test_resume_does_not_recover_temporary_metadata_from_another_run(tiny_run):
    args, pairs = tiny_run
    pipeline.run(args, pairs)
    sample = args.save_path / args.note / "0"
    (sample / "metadata.json").rename(sample / "metadata.json.tmp")
    args.resume = True
    args.attack_alpha = 2
    with pytest.raises(ValueError, match="different inputs/settings"):
        pipeline.run(args, pairs)
    assert (sample / "metadata.json.tmp").is_file()
