"""Independent byte-level audit of a completed GPU equivalence run (CPU only)."""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--run", type=Path, required=True)
p.add_argument("--repository", type=Path, required=True)
p.add_argument("--original-snapshot", type=Path, help="Override original source location recorded in observations")
p.add_argument("--release", type=Path, help="Override release source location recorded in observations")
p.add_argument("--commit", default="ca57421e23ba12da13df54db330c9330730e19c4")
a = p.parse_args()
reports = [json.loads((a.run / mode / "observations.json").read_text()) for mode in ["original", "cleaned"]]
summary = {"commit": a.commit, "source_manifest": {}, "arrays": {}, "pngs": {}, "samples": {}, "global_checks": {}}
summary["cleaned_source_manifest"] = {}
original_root = a.original_snapshot or Path(reports[0]["source_root"])
cleaned_root = a.release or Path(reports[1]["source_root"])


def sha(data):
    return hashlib.sha256(data).hexdigest()


blob = subprocess.check_output(
    ["git", "archive", a.commit, "attack.py", "dataloader", "utils", "methods", "explore"], cwd=a.repository
)
with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
    for name, digest in reports[0]["source_python_sha256"].items():
        original_bytes = archive.extractfile(name).read()
        snapshot_bytes = (original_root / name).read_bytes()
        summary["source_manifest"][name] = digest == sha(original_bytes) == sha(snapshot_bytes)
for name, digest in reports[1]["source_python_sha256"].items():
    summary["cleaned_source_manifest"][name] = digest == sha((cleaned_root / name).read_bytes())
for key in ["ids", "steps", "sources", "packages", "gpu", "cuda", "model_revisions", "milestones", "final_rng"]:
    summary["global_checks"][key] = reports[0][key] == reports[1][key]
for image_id in reports[0]["ids"]:
    samples = [r["samples"][image_id] for r in reports]
    checks = {
        key: samples[0][key] == samples[1][key]
        for key in [
            "caption",
            "all_captions",
            "schedule",
            "input_shape",
            "input_stride",
            "input_sha256",
            "rng_before_optimization",
            "rng_after_optimization",
            "crop_params",
            "crops",
        ]
    }
    checks["loss_bits"] = (
        np.asarray(samples[0]["losses"], dtype=np.float64).tobytes()
        == np.asarray(samples[1]["losses"], dtype=np.float64).tobytes()
    )
    checks["all_record_counts"] = all(
        len(s[k]) == reports[0]["steps"] for s in samples for k in ["losses", "crop_params", "crops"]
    )
    summary["samples"][image_id] = checks
    for kind in ["input", "attention", "clean", "adversarial"]:
        files = [a.run / mode / f"{image_id}_{kind}.npy" for mode in ["original", "cleaned"]]
        arrays = [np.load(f) for f in files]
        summary["arrays"][f"{image_id}_{kind}"] = {
            "dtype": [str(x.dtype) for x in arrays],
            "shape": [list(x.shape) for x in arrays],
            "payload_sha256": [sha(x.tobytes(order="C")) for x in arrays],
            "file_sha256": [sha(f.read_bytes()) for f in files],
            "dtype_equal": arrays[0].dtype == arrays[1].dtype,
            "shape_equal": arrays[0].shape == arrays[1].shape,
            "payload_bitwise_equal": arrays[0].tobytes(order="C") == arrays[1].tobytes(order="C"),
            "file_bitwise_equal": files[0].read_bytes() == files[1].read_bytes(),
        }
    for kind in ["clean", "adversarial"]:
        files = [a.run / mode / f"{image_id}_{kind}.png" for mode in ["original", "cleaned"]]
        summary["pngs"][f"{image_id}_{kind}"] = {
            "sha256": [sha(f.read_bytes()) for f in files],
            "file_bitwise_equal": files[0].read_bytes() == files[1].read_bytes(),
        }
summary["all_checks_pass"] = (
    all(summary["source_manifest"].values())
    and all(summary["cleaned_source_manifest"].values())
    and all(summary["global_checks"].values())
    and all(all(v.values()) for v in summary["samples"].values())
    and all(
        all(v[k] for k in ["dtype_equal", "shape_equal", "payload_bitwise_equal", "file_bitwise_equal"])
        for v in summary["arrays"].values()
    )
    and all(v["file_bitwise_equal"] for v in summary["pngs"].values())
)
(a.run / "independent_bitwise_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
print(
    json.dumps(
        {
            "all_checks_pass": summary["all_checks_pass"],
            "source_files": len(summary["source_manifest"]),
            "samples": list(summary["samples"]),
            "arrays": len(summary["arrays"]),
            "pngs": len(summary["pngs"]),
        },
        indent=2,
    )
)
raise SystemExit(0 if summary["all_checks_pass"] else 1)
