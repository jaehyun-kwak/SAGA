#!/usr/bin/env python3
"""Real original-versus-release GPU comparison; run workers ONLY inside Slurm.

Each mode is a separate fresh process. Original invokes unmodified OurAttack's
full image call path; cleaned invokes saga.pipeline.run. Observers record outputs
without replacing algorithms or seeding within an image. Explicit identical
image IDs isolate the algorithm from dataset subset selection. No AST extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import pickle
import platform
import random
import sys
import time
from pathlib import Path


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["original", "cleaned", "compare"], required=True)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ids", nargs="+", default=["0"])
    p.add_argument("--steps", type=int, default=300)
    args = p.parse_args()
    # Workers change directories; resolve caller-relative paths before that.
    args.original = args.original.expanduser().resolve()
    args.release = args.release.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    return args


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def worker(opts):
    if not os.getenv("SLURM_JOB_ID"):
        raise RuntimeError("GPU workers must be launched through Slurm, not on a login node")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["MPLBACKEND"] = "Agg"
    from importlib.metadata import version

    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms
    from torchvision.transforms import functional as F

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Validation requires exactly one Slurm-assigned visible GPU")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    base = opts.output / opts.mode
    base.mkdir(parents=True, exist_ok=False)
    root = opts.original if opts.mode == "original" else opts.release
    sys.path.insert(0, str(root))
    os.chdir(root)
    captions = {}
    for line in (opts.release / "input/coco.jsonl").read_text().splitlines():
        item = json.loads(line)
        captions[Path(item["file_name"]).stem] = item["caption"]
    pairs = [(opts.release / "input/images/aadcd" / (i + ".png"), captions[i]) for i in opts.ids]
    from huggingface_hub.constants import HF_HUB_CACHE

    checkpoints = [
        "openai/clip-vit-base-patch16",
        "openai/clip-vit-base-patch32",
        "laion/CLIP-ViT-G-14-laion2B-s12B-b42K",
        "Qwen/Qwen3-VL-8B-Instruct",
    ]
    revisions = {}
    for checkpoint in checkpoints:
        model_dir = Path(HF_HUB_CACHE) / ("models--" + checkpoint.replace("/", "--"))
        revisions[checkpoint] = {
            "main": (model_dir / "refs/main").read_text().strip() if (model_dir / "refs/main").exists() else None,
            "available_snapshots": sorted(p.name for p in (model_dir / "snapshots").iterdir())
            if (model_dir / "snapshots").exists()
            else [],
        }
    code_paths = [root / "attack.py"]
    for directory in ["methods", "explore", "utils", "dataloader"] if opts.mode == "original" else ["saga"]:
        code_paths.extend((root / directory).rglob("*.py"))
    code_hashes = {
        str(path.relative_to(root)): sha_bytes(path.read_bytes())
        for path in sorted(code_paths)
        if not any(part.startswith(".") for part in path.relative_to(root).parts)
    }
    report = {
        "mode": opts.mode,
        "steps": opts.steps,
        "ids": opts.ids,
        "python": platform.python_version(),
        "host": platform.node(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "packages": {p: version(p) for p in ["torch", "torchvision", "transformers", "qwen-vl-utils", "numpy"]},
        "strategy": "actual OurAttack.__call__" if opts.mode == "original" else "actual saga.pipeline.run",
        "source_root": str(root),
        "source_python_sha256": code_hashes,
        "hf_hub_cache": HF_HUB_CACHE,
        "model_revisions": revisions,
        "original_logging": "original init_wandb under WANDB_MODE=disabled; no remote uploads"
        if opts.mode == "original"
        else "release local logging",
        "input_pairing": "explicit same ordered IDs, no dataset subset sampling",
        "sources": [
            {"id": path.stem, "sha256": sha_bytes(path.read_bytes()), "target": target} for path, target in pairs
        ],
        "milestones": {},
        "samples": {},
    }
    current = {"id": None, "crops": [], "crop_params": [], "losses": []}

    def rng():
        return {
            "python": sha_bytes(pickle.dumps(random.getstate())),
            "numpy": sha_bytes(pickle.dumps(np.random.get_state())),
            "torch_cpu": sha_bytes(torch.get_rng_state().numpy().tobytes()),
            "torch_cuda": [sha_bytes(x.cpu().numpy().tobytes()) for x in torch.cuda.get_rng_state_all()],
        }

    def tensor_hash(t):
        return sha_bytes(t.detach().cpu().contiguous().numpy().tobytes())

    get_params = transforms.RandomResizedCrop.get_params

    def observed_params(img, scale, ratio):
        params = get_params(img, scale, ratio)
        if current["id"] is not None:
            current["crop_params"].append({"input_shape": list(img.shape), "params": list(params)})
        return params

    transforms.RandomResizedCrop.get_params = staticmethod(observed_params)

    def observe_crop(cls):
        original_forward = cls.forward

        def forward(self, image):
            crop, zone = original_forward(self, image)
            if current["id"] is not None:
                current["crops"].append(
                    {
                        "region": [self.top, self.left, self.height, self.width],
                        "zone": zone,
                        "sha256": tensor_hash(crop),
                        "rng_after": rng(),
                    }
                )
            return crop, zone

        cls.forward = forward

    def begin(image_id, tensor, schedule):
        current.update(id=image_id, crops=[], crop_params=[], losses=[])
        sample = {
            "schedule": schedule,
            "input_shape": list(tensor.shape),
            "input_stride": list(tensor.stride()),
            "input_sha256": tensor_hash(tensor),
            "rng_before_optimization": rng(),
        }
        report["samples"].setdefault(image_id, {}).update(sample)
        np.save(base / (image_id + "_input.npy"), tensor.detach().cpu().numpy())

    def finish(image_id, clean, adv, losses):
        sample = report["samples"][image_id]
        sample.update(
            {
                "rng_after_optimization": rng(),
                "losses": list(losses),
                "crop_params": current["crop_params"],
                "crops": current["crops"],
                "max_abs_delta_0_255": float(((adv - clean) * 255).abs().max().item()),
            }
        )
        for name, tensor in [("clean", clean), ("adversarial", adv)]:
            array = tensor.detach().cpu().numpy()
            np.save(base / (image_id + "_" + name + ".npy"), array)
            F.to_pil_image(tensor.detach().squeeze(0).cpu()).save(base / (image_id + "_" + name + ".png"))
        current["id"] = None
        write_json(base / "observations.json", report)

    started = time.perf_counter()
    if opts.mode == "original":
        # Use the original parser to retain every default outside supplied main flags.
        original_entry = importlib.import_module("attack")
        saved_argv = sys.argv
        sys.argv = [
            "attack",
            "--method",
            "ours",
            "--dataset",
            "aadcd",
            "--seed",
            "43",
            "--gpu",
            "0",
            "--num_images",
            str(len(pairs)),
            "--one_to_one",
            "--vlm_attention_extractor",
            "qwen3_vl",
            "--layer_idx",
            "29",
            "--use_content_words",
            "--no-use_smart_resize",
            "--progressive_mode",
            "--progressive_num_phases",
            "10",
            "--hotspot_top_k",
            "3",
            "--hotspot_iou_threshold",
            "0.3",
            "--hotspot_scale_min",
            "0.5",
            "--hotspot_scale_max",
            "0.9",
            "--hotspot_prob",
            "0.5",
            "--attack_steps",
            str(opts.steps),
            "--attack_alpha",
            "1",
            "--attack_epsilon",
            "16",
        ]
        args = original_entry.parse_arguments()
        sys.argv = saved_argv
        from methods.ours import OurAttack
        from utils.utils import set_seed

        crop_module = importlib.import_module("methods.ours.utils")
        attention_module = importlib.import_module("methods.ours.hotspot_method.metrics.open_vlm_attention")
        runner = importlib.import_module("methods.ours.hotspot_method.runner")
        optimization = importlib.import_module("methods.ours.attack_method.progressive_hotspot_attack")
        observe_crop(crop_module.RegionBiasedRandomResizedCrop)
        attention_fn = attention_module.VLMAttentionExtractor.get_attention_map
        attention_id = {"id": None}

        def observed_attention(self, image):
            result = attention_fn(self, image)
            image_id = attention_id["id"]
            np.save(base / (image_id + "_attention.npy"), result)
            report["samples"].setdefault(image_id, {}).update(
                {"caption": self.generated_captions[0], "all_captions": self.generated_captions}
            )
            return result

        attention_module.VLMAttentionExtractor.get_attention_map = observed_attention
        original_optimizer = runner.progressive_hotspot_attack

        def observed_optimizer(*a, **kw):
            image_id = Path(kw["image_path"]).stem
            begin(image_id, kw["image_org"], kw["hotspot_schedule"])
            adv = original_optimizer(*a, **kw)
            clean = (
                F.resize(
                    kw["image_org"].to(kw["cfg"].model.device),
                    [224, 224],
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                )
                .div(255)
                .clamp(0, 1)
            )
            finish(image_id, clean, adv, current["losses"])
            return adv

        runner.progressive_hotspot_attack = observed_optimizer
        original_log = optimization.log_losses

        def observed_loss(*a, **kw):
            current["losses"].append(kw["losses"]["similarity"])
            return original_log(*a, **kw)

        optimization.log_losses = observed_loss
        set_seed(args.seed)
        # Original CLI initializes W&B before OurAttack resets RNG to 2023.
        # Disabled mode still exercises original visualization/logging call sites.
        original_entry.init_wandb(args)
        attack = OurAttack(**vars(args))
        report["milestones"]["after_surrogate_loading"] = rng()
        for path, target in pairs:
            attention_id["id"] = path.stem
            with Image.open(path) as source:
                image = source.convert("RGB")
            attack(args, image, str(path), target)
    else:
        from saga import pipeline
        from saga.cli import parse_args
        from saga.crops import RegionBiasedRandomResizedCrop

        observe_crop(RegionBiasedRandomResizedCrop)
        args = parse_args(
            [
                "--num_images",
                str(len(pairs)),
                "--save_path",
                str(base / "pipeline_outputs"),
                "--note",
                "gpu_equivalence",
                "--attack_steps",
                str(opts.steps),
                "--seed",
                "43",
            ]
        )
        surrogate_loader = pipeline.load_surrogates

        def observed_loader(*a, **kw):
            result = surrogate_loader(*a, **kw)
            report["milestones"]["after_surrogate_loading"] = rng()
            return result

        pipeline.load_surrogates = observed_loader
        attention_fn = pipeline.extract_attention
        index = {"value": 0}

        def observed_attention(*a, **kw):
            result = attention_fn(*a, **kw)
            image_id = pairs[index["value"]][0].stem
            np.save(base / (image_id + "_attention.npy"), result[0])
            report["samples"].setdefault(image_id, {}).update({"caption": result[1], "all_captions": [result[1]]})
            return result

        pipeline.extract_attention = observed_attention
        original_optimizer = pipeline.optimize

        def observed_optimizer(image, target, schedule, *a, **kw):
            image_id = pairs[index["value"]][0].stem
            begin(image_id, image, schedule)
            result = original_optimizer(image, target, schedule, *a, **kw)
            finish(image_id, *result)
            index["value"] += 1
            return result

        pipeline.optimize = observed_optimizer
        pipeline.run(args, pairs)
    report["elapsed_sec"] = time.perf_counter() - started
    report["memory_bytes"] = {
        "max_allocated": torch.cuda.max_memory_allocated(0),
        "max_reserved": torch.cuda.max_memory_reserved(0),
    }
    report["final_rng"] = rng()
    write_json(base / "observations.json", report)
    print(json.dumps({"mode": opts.mode, "complete": True, "output": str(base), "elapsed_sec": report["elapsed_sec"]}))


def compare(opts):
    import numpy as np

    aroot, broot = [opts.output / x for x in ("original", "cleaned")]
    a, b = [json.loads((r / "observations.json").read_text()) for r in (aroot, broot)]
    result = {"exact_equal": True, "checks": {}, "samples": {}}
    for field in ["steps", "ids", "sources", "gpu", "cuda", "packages", "model_revisions", "milestones", "final_rng"]:
        same = a[field] == b[field]
        result["checks"][field] = same
        result["exact_equal"] &= same
    for image_id in opts.ids:
        sa, sb = a["samples"][image_id], b["samples"][image_id]
        checks = {}
        for field in [
            "caption",
            "all_captions",
            "schedule",
            "input_shape",
            "input_stride",
            "input_sha256",
            "rng_before_optimization",
            "rng_after_optimization",
            "losses",
            "crop_params",
            "crops",
        ]:
            checks[field] = sa[field] == sb[field]
        checks["loss_count_300_or_requested"] = len(sa["losses"]) == len(sb["losses"]) == opts.steps
        detail = {}
        for field in ["attention", "input", "clean", "adversarial"]:
            aa, bb = [np.load(r / (image_id + "_" + field + ".npy")) for r in (aroot, broot)]
            checks[field + "_tensor"] = bool(np.array_equal(aa, bb))
            detail[field] = {
                "shape_a": list(aa.shape),
                "shape_b": list(bb.shape),
                "max_abs_diff": float(np.max(np.abs(aa.astype(np.float64) - bb.astype(np.float64))))
                if aa.shape == bb.shape
                else None,
            }
        for field in ["clean", "adversarial"]:
            checks[field + "_png_bytes"] = (aroot / (image_id + "_" + field + ".png")).read_bytes() == (
                broot / (image_id + "_" + field + ".png")
            ).read_bytes()
        for field in ["losses", "crop_params", "crops"]:
            detail["first_different_" + field] = next(
                (i for i, (x, y) in enumerate(zip(sa[field], sb[field])) if x != y), None
            )
        result["samples"][image_id] = {"checks": checks, "details": detail}
        result["exact_equal"] &= all(checks.values())
    write_json(opts.output / "comparison.json", result)
    print(json.dumps(result, indent=2))
    if not result["exact_equal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    opts = arguments()
    if opts.mode == "compare":
        compare(opts)
    else:
        worker(opts)
