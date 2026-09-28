import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from loguru import logger

logger.remove()
parser = argparse.ArgumentParser(description="Synthetic CPU attention equivalence against immutable git source")
parser.add_argument("--original", type=Path, required=True)
parser.add_argument("--original-ref", default="ca57421")
parser.add_argument("--candidate", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
OLD = args.original.resolve()
NEW = args.candidate.resolve() / "saga"
commit = subprocess.check_output(
    ["git", "-C", str(OLD), "rev-parse", args.original_ref + "^{commit}"], text=True
).strip()
source_hashes = {}


def source(path):
    if path.is_relative_to(OLD):
        relative = path.relative_to(OLD)
        content = subprocess.check_output(["git", "-C", str(OLD), "show", f"{commit}:{relative}"])
        label = "original/" + str(relative)
    else:
        content = path.read_bytes()
        label = "candidate/" + str(path.relative_to(NEW.parent))
    source_hashes[label] = hashlib.sha256(content).hexdigest()
    return content.decode("utf-8")


def extracted(path, names, env):
    tree = ast.parse(source(path))
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.Assign))
        and (
            (isinstance(n, ast.FunctionDef) and n.name in names)
            or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "STOP_WORDS" for t in n.targets))
        )
    ]
    exec(  # noqa: S102 - execute locally selected, fingerprinted reference functions
        compile(
            "from __future__ import annotations\n" + ast.unparse(ast.Module(body=nodes, type_ignores=[])),
            str(path),
            "exec",
        ),
        env,
    )


def setup(root, new=False):
    import string

    env = {"torch": torch, "np": np, "logger": logger, "string": string, "DEBUG": False, "DBEUG": False}
    names = [
        "get_token_indices",
        "aggregate_llm_attention",
        "heterogenous_stack",
        "aggregate_prompt_attention",
        "build_attention_matrix",
        "get_token_img_shape",
    ]
    extracted(root / ("attention_utils.py" if new else "explore/attention/utils.py"), names, env)
    extracted(
        root / ("text.py" if new else "explore/attention/utils_text.py"),
        ["clean_token", "is_content_word", "filter_generated_tokens"],
        env,
    )
    extracted(
        root / ("attention_map.py" if new else "explore/attention/attention_map.py"),
        ["_canonical_norm_mode", "get_attention_map_from_generated_text_vlm"],
        env,
    )
    return env


old, new = setup(OLD), setup(NEW, True)
assert old["STOP_WORDS"] == new["STOP_WORDS"]


class Output(dict):
    __getattr__ = dict.__getitem__


class Tokenizer:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, text, **kw):
        return {"input_ids": [list(range(2 if text == "prefix" else 9))]}

    def decode(self, ids, **kw):
        tokens = (
            ["the", "red", "cat", ".", "<|im_end|>"]
            if self.mode == "mixed"
            else ["the"] * 5
            if self.mode == "stop"
            else ["cat"] * 5
        )
        return " ".join(tokens[int(i) - 20] for i in ids)


class Model:
    def __init__(self, output):
        self.output = output

    def generate(self, **kw):
        self.kw = kw
        return self.output


count = 0
cases = []
for dtype in [torch.float32, torch.bfloat16, torch.float16]:
    for layer in [0, 29, None]:
        for mode in ["mixed", "stop", "content"]:
            for shape_case in ["random", "uniform", "zero_vision"]:
                torch.manual_seed(24)
                steps = []
                for step in range(5):
                    a = torch.rand(1, 2, 12 if step == 0 else 1, 12 + step, dtype=dtype)
                    if shape_case == "uniform":
                        a.fill_(1)
                    if shape_case == "zero_vision":
                        a[:, :, :, 3:9] = 0
                    a = a / a.sum(-1, keepdim=True)
                    steps.append(tuple(a.clone() for _ in range(30)))
                output = Output(
                    sequences=torch.tensor([list(range(12)) + list(range(20, 25))]), attentions=tuple(steps)
                )
                inputs = {"input_ids": torch.arange(12)[None, :], "image_grid_thw": torch.tensor([[1, 4, 6]])}
                processor = SimpleNamespace(
                    tokenizer=Tokenizer(mode),
                    image_processor=SimpleNamespace(merge_size=2),
                    batch_decode=lambda *a, **kw: ["prefix<|vision_start|>vision<|vision_end|>suffix"],
                )
                m1, m2 = Model(output), Model(output)
                a, c1 = old["get_attention_map_from_generated_text_vlm"](
                    m1, processor, inputs, "qwen3_vl", layer_idx=layer, use_content_words=True
                )
                b, c2 = new["get_attention_map_from_generated_text_vlm"](
                    m2, processor, inputs, "qwen3_vl", layer_idx=layer, use_content_words=True
                )
                assert torch.equal(a, b), (dtype, layer, mode, shape_case, a, b)
                assert c1 == c2
                for k in m1.kw:
                    if k not in inputs:
                        assert m1.kw[k] == m2.kw[k]
                # Compare extractor wrapper's two old renormalizations vs release's one.
                oa = np.mean([a], axis=0)

                def norm(x):
                    mn, mx = x.min(), x.max()
                    return (x - mn) / (mx - mn) if mx > mn else np.ones_like(x) * 0.5

                oa = norm(norm(oa))
                nb = norm(b.detach().float().cpu().numpy())
                assert np.array_equal(oa, nb)
                count += 1
                cases.append(
                    {
                        "dtype": str(dtype),
                        "layer": layer,
                        "tokens": mode,
                        "attention": shape_case,
                        "attention_exact": True,
                        "caption_exact": True,
                        "generation_settings_exact": True,
                        "wrapper_normalization_exact": True,
                    }
                )
result = {
    "passed": True,
    "cases_count": count,
    "cases": cases,
    "original_repo": str(OLD),
    "original_ref": args.original_ref,
    "original_commit": commit,
    "candidate": str(NEW.parent),
    "source_sha256": source_hashes,
    "stop_words_count": len(old["STOP_WORDS"]),
    "stop_words_exact": True,
    "torch_version": torch.__version__,
    "numpy_version": np.__version__,
    "method": "Unmodified function ASTs from immutable git show and candidate source; synthetic generation output",
    "scope": "Qwen3-VL generation/minmax content-word extraction; CPU synthetic, no model loading",
    "wrapper_check": "Two original minmax normalizations versus one candidate normalization computed algebraically",
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(f"PASS {count} exact attention cases, {len(old['STOP_WORDS'])} identical stop words; {args.output}")
