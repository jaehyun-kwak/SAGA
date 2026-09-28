"""Protect executed main-method behavior, including research-code quirks."""

from collections import Counter

import numpy as np
import torch

import saga.optimization as optimization
from saga.hotspots import build_schedule, compute_box_iou, get_top_k_boxes_fixed_area


def test_equal_scores_keep_candidate_enumeration_order():
    # Stable score sorting preserves height -> row -> column on exact ties.
    boxes = get_top_k_boxes_fixed_area(np.zeros((4, 4)), 0.25, k=3, iou_threshold=1.0)
    assert boxes == [(0, 1, 0, 1), (1, 2, 0, 1), (2, 3, 0, 1)]


def test_iou_fallback_really_permits_overlapping_regions():
    boxes = get_top_k_boxes_fixed_area(np.zeros((14, 14)), 0.9, k=3, iou_threshold=0.3)
    assert len(boxes) == 3
    assert len(set(boxes)) == 3
    assert any(compute_box_iou(boxes[0], other) > 0.3 for other in boxes[1:])
    assert get_top_k_boxes_fixed_area(np.ones((14, 14)), 1.0, k=3) == [(0, 13, 0, 13)]


def test_main_uses_hotspot_every_step_and_last_region_gets_remainder(monkeypatch):
    seen = []

    class CropSpy:
        def __init__(self, *, size, scale, hotspot_rect, hotspot_prob):
            assert hotspot_prob == 1.0
            seen.append(hotspot_rect)

        def __call__(self, image):
            return image, "hotspot"

    class Objective:
        def set_ground_truth(self, target, kind):
            assert (target, kind) == ("target", "text")

        def __call__(self, features):
            return features

    class Quiet:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **kwargs):
            pass

    monkeypatch.setattr(optimization, "RegionBiasedRandomResizedCrop", CropSpy)
    monkeypatch.setattr(optimization, "tqdm", Quiet)
    # Keep this scheduling regression cheap without changing crop allocation.
    monkeypatch.setattr(optimization.F, "resize", lambda image, *args, **kwargs: image)
    schedule = build_schedule(np.ones((14, 14), dtype=np.float32))
    clean, adversarial, losses = optimization.optimize(
        torch.full((1, 3, 2, 2), 100.0),
        "target",
        schedule,
        lambda image, kind: image.mean(),
        Objective(),
        device="cpu",
        steps=300,
        epsilon=16.0,
    )
    rects = [
        (int(t * 224), int(left * 224), max(1, int(h * 224)), max(1, int(w * 224)))
        for t, left, h, w in (item["box"] for item in schedule)
    ]
    assert seen == [rect for rect in rects[:-1] for _ in range(10)] + [rects[-1]] * 30
    assert Counter(seen)[rects[-1]] == 30
    assert len(losses) == 300
    torch.testing.assert_close((adversarial - clean) * 255, torch.full_like(clean, 16.0))
