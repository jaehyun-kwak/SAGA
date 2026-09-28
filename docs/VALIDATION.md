# Strict equivalence validation

Reference: `Review-Generation-Attack` commit
`ca57421e23ba12da13df54db330c9330730e19c4`.

The released main command matches the current reference's Qwen3-VL layer 29,
generation-time clean attention, content-word/minmax aggregation, fixed progressive
SAGA ordering, B16/B32/Laion ensemble, 300-step alpha 1 / epsilon 16 configuration.
Operational features such as strict input validation, sample-ID output paths and
explicit resume remain release conveniences. Non-main ablations are not included.

## Output simplification (2026-09-28)

The public pipeline now writes only `original.png`, `adversarial.png`, and
`metadata.json` per sample. Run settings and resume checks are stored in metadata;
no root `run.json` or per-sample attention/schedule/loss files are written.
The GPU evidence below describes the earlier validated snapshot. The attack
computation is unchanged, but those recorded source hashes are not a fingerprint
of the current output-handling code.

## Corrections after the initial server-test archive

- Restored contiguous CHW tensors and the original default floating dtype.
- Restored sampling from unsorted `os.listdir()` followed by selected-ID sorting.
- Restored Qwen `device_map="auto"`, inputs on `model.device`, and processor pixel defaults.
- Restored the separate original seed 43 initializer and seed 2023 model initializer,
  including exact explicit CUDA seed calls and `PYTHONHASHSEED` assignment.
- Removed an extra `torch.cuda.set_device` call from the released pipeline.

Subset selection was compared with the real original loader in the SAME input
directory at counts 1, 10, 1000; all ID/target pairs matched. Different filesystems
can enumerate the same filenames differently, as they can in the original code.
All 1,002 source images/TXT/JSONL files are unchanged from the original archive.

## CPU checks

- 18 release unit/regression tests passed; Ruff passed.
- 81 attention differential cases match exactly (including dtype, token-filter and layer cases).
- 640 rectangle-selector and 64 full-schedule differential cases match exactly.
- Three nonlinear-toy optimizer cases (60/7, 300/28, 31/30 steps/schedule items) have
  bit-identical output tensors and matching Python/Torch RNG states.
- Crop class, three surrogate forward methods and ensemble objective match the
  reference executable ASTs in their released main path.

The release tests were run in the lockfile environment using Python 3.12.3,
PyTorch 2.8.0+cu126, torchvision 0.23.0+cu126, Transformers 4.57.1 and NumPy 2.2.4.
CPU comparison JSON files additionally record their own execution context.

## Actual-model GPU comparison

**PASS**: Slurm job `32501` completed with exit code 0 on `srv01`, NVIDIA RTX A5000.
The two workers ran separately in the same pinned environment and used the same
cached checkpoint revisions. Image IDs `0` and `1` were each optimized for 300
steps in sequence, without resetting the RNG between images.

Both workers selected 22 actual schedule items per image. All 600 per-step losses,
captions, attention maps, schedules, crop parameters/tensor hashes, input strides
and before/after/final RNG states match exactly. An independent agent verified
all 8 corresponding NPY files (dtype, shape, payload and complete file bytes) and
all 4 PNG file pairs are byte-identical, including signed-zero representations.
The original 128 observed Python file hashes match the immutable reference commit,
and the cleaned worker's source hashes match the distributed attack modules.

Evidence is in `docs/validation/gpu/comparison.json`,
`docs/validation/gpu/independent_bitwise_audit.json` and each worker's observations,
arrays and images. The standalone audit tool also checks source provenance.

The original worker took 253.24 seconds and the cleaned worker 167.41 seconds.
These timings include different cache warmth and observation/logging overhead;
they are not a speed comparison. Automatic Qwen placement used CPU offload on
this 24 GB GPU, preserving the original loader behavior. Maximum CUDA reserved
memory was about 22.09 GiB (allocated 21.60 GiB); see each worker report for bytes.

`tools/gpu_equivalence.py` uses real `OurAttack.__call__` and real `saga.pipeline.run`,
not extracted/reimplemented attack algorithms. Separate processes on the same
Slurm-assigned GPU use explicit identical IDs in the same order. The original's
W&B path runs in disabled mode, without uploads. Observers record captions,
attention maps, schedules, crop parameters/tensor hashes, per-step losses,
input/clean/adversarial tensors, PNG bytes and RNG states. Model revisions,
source hashes, packages and peak CUDA memory are also captured.

## Re-run the checks

From this release directory, with the original Git checkout available:

```bash
uv sync --frozen
uv run --frozen python -m pytest -q
uv run --frozen ruff check saga tests tools attack.py
uv run --frozen python tools/cpu_equivalence.py \
  --original /path/to/Review-Generation-Attack --original-ref ca57421 \
  --candidate "$PWD" --output /tmp/saga-cpu-equivalence.json
uv run --frozen python tools/attention_equivalence.py \
  --original /path/to/Review-Generation-Attack --original-ref ca57421 \
  --candidate "$PWD" --output /tmp/saga-attention-equivalence.json
```

The original GPU path also imports its original-only dependencies (e.g. W&B,
OmegaConf, matplotlib and pandas). Use the same pinned runtime plus those
reference dependencies for BOTH workers; no such dependencies are required by
the standalone released attack. The exact reference-only installation and runtime
constraints are recorded in `docs/validation/reference-requirements.txt` and
`runtime-constraints.txt`. Run the three modes sequentially inside a
Slurm allocation with one visible CUDA GPU and cached checkpoints:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled PYTHONHASHSEED=2023
for mode in original cleaned compare; do
  python tools/gpu_equivalence.py --mode "$mode" \
    --original /path/to/Review-Generation-Attack --release "$PWD" \
    --output /tmp/saga-gpu-equivalence --ids 0 1 --steps 300
done
```

These checks concern the current source commit. They do not recover the historical
260120 launch commit/environment or establish the ASR of a full 1,000-image run.

## Included historical attacked images

The release also includes 1,000 saved adversarial PNGs from
`260120_aadcd_ours_qwen3vl_seed43`. All saved original RGB pixel arrays match the
bundled clean inputs, all source directory names match the exact JSONL targets,
and every attacked PNG is copied byte-for-byte. See
`attacked_images/README.md`, `attacked_images/targets.jsonl`, and
`docs/validation/attacked_images_manifest.json`. This is provenance validation
of historical artifacts, separate from the current-code two-image GPU smoke.
