# -*- coding: utf-8 -*-
"""Unit tests that do not require model files."""
import numpy as np
import pytest

from ppocrv6_min.det import DBPostProcess
from ppocrv6_min.io import imread_unicode
from ppocrv6_min.pipeline import order_quad, warp_quad
from ppocrv6_min.rec import load_dict


def test_load_dict_strips_quotes(tmp_path):
    yml = tmp_path / "inference.yml"
    yml.write_text(
        "PostProcess:\n"
        "  name: CTCLabelDecode\n"
        "  character_dict:\n"
        "  - '!'\n"
        "  - '\"'\n"
        "  - ''''\n"
        "  - $\n"
        "  - 中\n"
        "  - 国\n",
        encoding="utf-8")
    chars = load_dict(str(yml))
    assert chars == ["!", '"', "'", "$", "中", "国"]


def test_load_dict_missing_key(tmp_path):
    yml = tmp_path / "inference.yml"
    yml.write_text("Global:\n  model_name: x\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_dict(str(yml))


def test_unclip_expands_polygon():
    post = DBPostProcess(unclip_ratio=2.0)
    box = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    expanded = post.unclip(box, 2.0)  # list of polygons; expect exactly one
    assert len(expanded) == 1
    pts = np.asarray(expanded[0])
    assert pts.shape[0] >= 4
    xs = pts[:, 0]
    ys = pts[:, 1]
    assert xs.max() > 10 and xs.min() < 0
    assert ys.max() > 10 and ys.min() < 0


def test_order_quad():
    # feed points in a shuffled order, expect tl, tr, br, bl back
    shuffled = np.array([[100, 100], [0, 0], [100, 0], [0, 100]],
                        dtype=np.float32)
    q = order_quad(shuffled)
    assert tuple(q[0]) == (0, 0)          # tl
    assert tuple(q[1]) == (100, 0)        # tr
    assert tuple(q[2]) == (100, 100)      # br
    assert tuple(q[3]) == (0, 100)        # bl


def test_warp_quad_horizontal():
    img = np.zeros((40, 200, 3), dtype=np.uint8)
    img[:, :, 0] = 255
    quad = np.array([[20, 10], [180, 10], [180, 30], [20, 30]],
                    dtype=np.float32)
    strip = warp_quad(img, quad)
    assert strip is not None
    assert strip.shape[1] >= 160  # width ~ 160
    assert strip.shape[0] >= 8


def test_imread_unicode_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        imread_unicode(str(tmp_path / "nope.png"))
