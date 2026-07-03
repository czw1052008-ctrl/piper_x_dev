from picking_perception.segmentation import segment_blueberry_hsv
import numpy as np


def test_segment_blueberry_hsv_detects_purple():
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    rgb[40:60, 40:60] = [60, 40, 140]  # dark purple berry (common under indoor light)
    mask = segment_blueberry_hsv(rgb)
    assert mask[50, 50] == 1
