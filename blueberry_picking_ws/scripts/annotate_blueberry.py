#!/usr/bin/env python3
"""Local YOLO bbox annotator for blueberries (class 0 = blueberry).

Usage:
  python3 scripts/annotate_blueberry.py
  python3 scripts/annotate_blueberry.py --images datasets/blueberry/images

Workflow (faster):
  bash scripts/run_prelabel_blueberry.sh --overwrite
  bash scripts/run_annotate_blueberry.sh    # review / fix / confirm

Controls:
  Mouse drag     Draw new bbox
  Click box      Select box (yellow)
  x / Del        Delete selected box
  c              Clear all boxes on this image
  u              Undo last box
  s              Save labels and next image
  a / Left       Previous image
  d / Right      Next image (auto-save)
  q / Esc        Quit
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

CLASS_ID = 0
CLASS_NAME = 'blueberry'
WIN = 'annotate_blueberry'


@dataclass
class Box:
    x0: int
    y0: int
    x1: int
    y1: int

    def normalized(self, w: int, h: int) -> Tuple[float, float, float, float]:
        x0, x1 = sorted((self.x0, self.x1))
        y0, y1 = sorted((self.y0, self.y1))
        cx = (x0 + x1) * 0.5 / w
        cy = (y0 + y1) * 0.5 / h
        bw = max(x1 - x0, 1) / w
        bh = max(y1 - y0, 1) / h
        return cx, cy, bw, bh

    def draw(self, img: np.ndarray, color=(0, 255, 0), thick: int = 2) -> None:
        x0, x1 = sorted((self.x0, self.x1))
        y0, y1 = sorted((self.y0, self.y1))
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thick)


class Annotator:
    def __init__(self, image_dir: str, label_dir: str) -> None:
        self.image_dir = image_dir
        self.label_dir = label_dir
        os.makedirs(self.label_dir, exist_ok=True)
        exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
        paths: List[str] = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(image_dir, ext)))
        self.paths = sorted(paths)
        if not self.paths:
            raise SystemExit(f'No images in {image_dir}')
        self.idx = 0
        self.boxes: List[Box] = []
        self._selected: int | None = None
        self._drag_start: Tuple[int, int] | None = None
        self._drag_end: Tuple[int, int] | None = None
        self._base: np.ndarray | None = None

    def _label_path(self, image_path: str) -> str:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(self.label_dir, f'{stem}.txt')

    def _load_boxes(self, image_path: str, w: int, h: int) -> List[Box]:
        lp = self._label_path(image_path)
        if not os.path.isfile(lp):
            return []
        boxes: List[Box] = []
        with open(lp, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                _, cx, cy, bw, bh = parts[:5]
                cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
                x0 = int((cx - bw / 2) * w)
                y0 = int((cy - bh / 2) * h)
                x1 = int((cx + bw / 2) * w)
                y1 = int((cy + bh / 2) * h)
                boxes.append(Box(x0, y0, x1, y1))
        return boxes

    def _save_boxes(self, image_path: str) -> None:
        if self._base is None:
            return
        h, w = self._base.shape[:2]
        lp = self._label_path(image_path)
        with open(lp, 'w', encoding='utf-8') as f:
            for b in self.boxes:
                cx, cy, bw, bh = b.normalized(w, h)
                f.write(f'{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')
        print(f'[save] {lp} ({len(self.boxes)} boxes)')

    def _load_image(self) -> None:
        path = self.paths[self.idx]
        bgr = cv2.imread(path)
        if bgr is None:
            raise SystemExit(f'Failed to read {path}')
        self._base = bgr
        h, w = bgr.shape[:2]
        self.boxes = self._load_boxes(path, w, h)
        self._selected = None
        self._drag_start = None
        self._drag_end = None
        if self.boxes:
            print(f'[load] {os.path.basename(path)}: {len(self.boxes)} pre-label(s)')

    def _hit_box(self, x: int, y: int) -> int | None:
        for i in range(len(self.boxes) - 1, -1, -1):
            x0, x1 = sorted((self.boxes[i].x0, self.boxes[i].x1))
            y0, y1 = sorted((self.boxes[i].y0, self.boxes[i].y1))
            if x0 <= x <= x1 and y0 <= y <= y1:
                return i
        return None

    def _delete_selected(self) -> None:
        if self._selected is None:
            return
        self.boxes.pop(self._selected)
        self._selected = None

    def _render(self) -> np.ndarray:
        assert self._base is not None
        vis = self._base.copy()
        for i, b in enumerate(self.boxes):
            color = (0, 255, 255) if i == self._selected else (0, 255, 0)
            thick = 3 if i == self._selected else 2
            b.draw(vis, color, thick)
        if self._drag_start and self._drag_end:
            Box(self._drag_start[0], self._drag_start[1],
                self._drag_end[0], self._drag_end[1]).draw(vis, (0, 200, 255), 2)
        title = f'[{self.idx + 1}/{len(self.paths)}] {os.path.basename(self.paths[self.idx])}'
        cv2.putText(vis, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            vis, 'review: click select | drag add | x del | c clear | s/d next | a prev | q quit',
            (8, vis.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        return vis

    def _on_mouse(self, event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drag_start = (x, y)
            self._drag_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self._drag_start:
            self._drag_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self._drag_start:
            self._drag_end = (x, y)
            x0, x1 = self._drag_start[0], self._drag_end[0]
            y0, y1 = self._drag_start[1], self._drag_end[1]
            if abs(x1 - x0) > 4 and abs(y1 - y0) > 4:
                self.boxes.append(Box(x0, y0, x1, y1))
                self._selected = len(self.boxes) - 1
            else:
                hit = self._hit_box(x, y)
                self._selected = hit
            self._drag_start = None
            self._drag_end = None

    def run(self) -> None:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WIN, self._on_mouse)
        self._load_image()
        while True:
            cv2.imshow(WIN, self._render())
            key = cv2.waitKey(20) & 0xFF
            if key in (ord('q'), 27):
                self._save_boxes(self.paths[self.idx])
                break
            if key in (ord('x'), 127, 8):  # x, delete, backspace
                self._delete_selected()
            if key == ord('c'):
                self.boxes.clear()
                self._selected = None
            if key == ord('u') and self.boxes:
                self.boxes.pop()
                self._selected = None
            if key in (ord('s'), ord('d'), 83):  # s, d, right arrow
                self._save_boxes(self.paths[self.idx])
                if self.idx < len(self.paths) - 1:
                    self.idx += 1
                    self._load_image()
                elif key == ord('s'):
                    print('[done] last image saved')
            if key in (ord('a'), 81) and self.idx > 0:  # a, left arrow
                self._save_boxes(self.paths[self.idx])
                self.idx -= 1
                self._load_image()
        cv2.destroyAllWindows()


def main() -> int:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--images', default=os.path.join(root, 'datasets', 'blueberry', 'images'))
    parser.add_argument('--labels', default=os.path.join(root, 'datasets', 'blueberry', 'labels'))
    args = parser.parse_args()
    Annotator(args.images, args.labels).run()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
