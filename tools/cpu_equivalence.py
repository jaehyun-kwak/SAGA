"""Model-free CPU differential audit against an immutable original Git revision.

Example:
  python cpu_equivalence.py --original /path/to/research/repo --candidate /path/to/SAGA

No downloads or production-source mutations. Original executable AST nodes are
compiled unchanged; only optional logging/artifact writers are replaced by noops.
Toy objectives validate optimization mechanics, not pretrained-model inference.
"""

import argparse
import ast
import importlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import functional as F


class Quiet:
    def __init__(self, iterable, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, *args, **kwargs):
        pass

    def write(self, *args, **kwargs):
        pass


def noop(*args, **kwargs):
    pass


class ToyExtractor:
    def __call__(self, image, kind):
        return (image / 100).sin().square().mean() + (image[..., ::3, ::5] / 70).cos().mean()


class ToyObjective:
    def set_ground_truth(self, *args):
        pass

    def __call__(self, features):
        return features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--original-ref", default="ca57421")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    original = args.original.resolve()
    candidate = args.candidate.resolve()
    ref = subprocess.check_output(["git", "-C", str(original), "rev-parse", args.original_ref], text=True).strip()

    def source(relative):
        return subprocess.check_output(["git", "-C", str(original), "show", f"{ref}:{relative}"], text=True)

    def extract(relative, names, namespace):
        nodes = [
            node
            for node in ast.parse(source(relative)).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
        ]
        assert {node.name for node in nodes} == set(names), (relative, names)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), relative, "exec"), namespace)  # noqa: S102 - audited original AST

    sys.path.insert(0, str(candidate))
    hotspots = importlib.import_module("saga.hotspots")
    optimization = importlib.import_module("saga.optimization")
    optimization.tqdm = Quiet
    torch.set_num_threads(1)
    report = {
        "original_commit": ref,
        "candidate": str(candidate),
        "selector_cases": 0,
        "schedule_cases": 0,
        "optimizer_cases": [],
        "ast_checks": {},
    }

    selector_file = "methods/ours/hotspot_method/metrics/open_vlm_attention.py"
    outer = next(
        node
        for node in ast.parse(source(selector_file)).body
        if isinstance(node, ast.FunctionDef) and node.name == "get_open_vlm_attention_rect_cells"
    )
    nodes = [
        node
        for node in outer.body
        if isinstance(node, ast.FunctionDef) and node.name in {"compute_box_iou", "get_top_k_boxes_fixed_area"}
    ]
    selector_ns = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), selector_file, "exec"), selector_ns)  # noqa: S102 - audited original AST

    rng = np.random.default_rng(0)
    for shape in [(1, 1), (1, 9), (2, 3), (7, 11), (14, 14), (16, 24), (28, 28), (4, 40)]:
        for dtype in [np.float32, np.float64]:
            for attention in [
                np.zeros(shape, dtype=dtype),
                np.ones(shape, dtype=dtype),
                rng.random(shape).astype(dtype),
                rng.integers(0, 3, shape).astype(dtype),
            ]:
                expected = []
                for i in range(10):
                    ratio = 0.0 + (1 - 0.0) * (i + 1) / 10
                    boxes = selector_ns["get_top_k_boxes_fixed_area"](attention, ratio, k=3, iou_threshold=0.3)
                    assert boxes == hotspots.get_top_k_boxes_fixed_area(attention, ratio, k=3, iou_threshold=0.3)
                    height, width = shape
                    for j, (xmin, xmax, ymin, ymax) in enumerate(boxes):
                        top = np.clip(ymin / height, 0.0, 1.0)
                        left = np.clip(xmin / width, 0.0, 1.0)
                        expected.append(
                            {
                                "threshold": ratio,
                                "box": (
                                    float(top),
                                    float(left),
                                    float(np.clip((ymax - ymin + 1) / height, 0.0, 1.0 - top)),
                                    float(np.clip((xmax - xmin + 1) / width, 0.0, 1.0 - left)),
                                ),
                                "phase_group": i + 1,
                                "sub_phase": j + 1,
                            }
                        )
                    report["selector_cases"] += 1
                try:
                    actual = hotspots.build_schedule(attention)
                except ValueError:
                    assert not expected
                else:
                    assert actual == expected
                report["schedule_cases"] += 1

    ns = {
        "torch": torch,
        "nn": nn,
        "transforms": transforms,
        "F": F,
        "random": random,
        "MainConfig": object,
        "time": time,
        "logger": NS(info=noop, warning=noop),
        "tqdm": Quiet,
    }
    extract("methods/ours/utils.py", ["RegionBiasedRandomResizedCrop"], ns)
    extract(
        "methods/ours/attack_method/progressive_hotspot_attack.py",
        ["_copy_schedule", "_apply_hotspot_ordering", "_schedule_to_pixel_rects", "progressive_hotspot_attack"],
        ns,
    )
    for name in [
        "_save_stage_checkpoint",
        "init_perturb_accum",
        "log_hotspot_box",
        "log_losses",
        "update_perturb_accum",
        "log_noise_heatmap",
        "_save_attention_cost",
        "_save_hotspot_trace",
        "_save_experiment_trace",
    ]:
        ns[name] = noop
    for seed, shape, steps, entries in [
        (9, (1, 3, 180, 251), 60, 7),
        (43, (1, 3, 224, 224), 300, 28),
        (2023, (1, 3, 85, 96), 31, 30),
    ]:
        torch.manual_seed(123)
        image = torch.rand(shape) * 255
        schedule = [
            {
                "box": (0.0, 0.0, 0.2 + 0.8 * i / (entries - 1), 0.2 + 0.8 * i / (entries - 1)),
                "threshold": (i + 1) / entries,
                "phase_group": i + 1,
                "sub_phase": 1,
            }
            for i in range(entries)
        ]
        original_args = NS(hotspot_scale_min=0.5, hotspot_scale_max=0.9)
        cfg = NS(
            model=NS(device="cpu"),
            optim=NS(steps=steps, alpha=1.0, epsilon=16.0),
            data=NS(source_type="image", target_type="text"),
        )
        random.seed(seed)
        torch.manual_seed(seed)
        old = ns["progressive_hotspot_attack"](
            cfg, original_args, ToyExtractor(), ToyObjective(), image, "synthetic", "target", schedule
        )
        old_rng = (random.getstate(), torch.get_rng_state())
        random.seed(seed)
        torch.manual_seed(seed)
        _, new, _ = optimization.optimize(
            image, "target", schedule, ToyExtractor(), ToyObjective(), device="cpu", steps=steps
        )
        rng_equal = old_rng[0] == random.getstate() and torch.equal(old_rng[1], torch.get_rng_state())
        assert torch.equal(old, new), (seed, (old - new).abs().max().item())
        assert rng_equal, seed
        report["optimizer_cases"].append(
            {
                "seed": seed,
                "shape": shape,
                "steps": steps,
                "entries": entries,
                "exact_tensor_equal": True,
                "rng_equal": True,
            }
        )

    def executable_ast(node):
        # Docstring-only changes have no effect on executed arithmetic.
        for child in ast.walk(node):
            if isinstance(child, (ast.FunctionDef, ast.ClassDef)) and child.body:
                first = child.body[0]
                if (
                    isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)
                ):
                    child.body = child.body[1:]
        return ast.dump(node, include_attributes=False)

    for old_name, new_name in [("ClipB16", "clipb16"), ("ClipB32", "clipb32"), ("ClipLaion", "cliplaion")]:
        old_tree = ast.parse(source(f"methods/ours/surrogates/FeatureExtractors/{old_name}.py"))
        new_tree = ast.parse((candidate / f"saga/surrogates/{new_name}.py").read_text())
        for method in ["forward", "encode_image", "encode_text"]:
            old_nodes = [
                node for node in ast.walk(old_tree) if isinstance(node, ast.FunctionDef) and node.name == method
            ]
            new_nodes = [
                node for node in ast.walk(new_tree) if isinstance(node, ast.FunctionDef) and node.name == method
            ]
            if old_nodes or new_nodes:
                assert len(old_nodes) == len(new_nodes) == 1
                equal = executable_ast(old_nodes[0]) == executable_ast(new_nodes[0])
                report["ast_checks"][f"{old_name}.{method}"] = equal
                assert equal, (old_name, method)
    for name, old_path, new_path in [
        ("EnsembleFeatureExtractor", "methods/ours/surrogates/FeatureExtractors/Base.py", "saga/surrogates/base.py"),
        ("EnsembleFeatureLoss", "methods/ours/surrogates/FeatureExtractors/Base.py", "saga/surrogates/base.py"),
        ("RegionBiasedRandomResizedCrop", "methods/ours/utils.py", "saga/crops.py"),
    ]:
        old = next(n for n in ast.parse(source(old_path)).body if isinstance(n, ast.ClassDef) and n.name == name)
        new = next(
            n
            for n in ast.parse((candidate / new_path).read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == name
        )
        equal = executable_ast(old) == executable_ast(new)
        report["ast_checks"][name] = equal
        assert equal, name
    report["status"] = "passed"
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.output:
        args.output.write_text(payload + "\n")


if __name__ == "__main__":
    main()
