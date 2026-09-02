#!/usr/bin/env python3
"""点果心即可。一下一颗。不必标枝/硬物/自身。

  左键          点果心
                腕部：SAM 贴这一颗（仍裁到半径内）
                全局：只盖一个小圆（不用 SAM，避免一点整簇）
  滚轮 / - =    圆半径（全局默认很小）
  z              全图缩放（不裁切，始终能看到整张）
  右下角         鼠标附近放大镜
  s              开关 SAM（全局默认关）
  Shift+左键 / e 擦除
  x              删这一颗
  c              清空本张
  u 撤销
  d / a          下一张 / 上一张（自动保存）
  q              保存退出
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'picking_perception'))

PIX_BERRY, PIX_BRANCH, PIX_RIGID, PIX_EGO = 1, 2, 3, 4
CLASS_NAMES = {1: 'berry', 2: 'branch', 3: 'rigid', 4: 'ego', 0: 'erase'}
TINT = {
    1: (255, 0, 255),
    2: (0, 220, 0),
    3: (0, 0, 255),
    4: (220, 200, 40),
}
YOLO_WEIGHTS = ROOT / 'runs' / 'detect' / 'blueberry-unified-1' / 'weights' / 'best.pt'
WIN = 'annotate_scene_seg'


def _list_images(data: Path) -> List[Path]:
    paths: List[Path] = []
    for split in ('train', 'val'):
        d = data / split / 'images'
        if not d.is_dir():
            continue
        paths.extend(sorted(
            p for p in d.iterdir()
            if p.suffix.lower() in ('.png', '.jpg', '.jpeg')))
    return paths


def _mask_path(img: Path) -> Path:
    return img.parent.parent / 'labels' / f'{img.stem}_mask.png'


def _txt_path(img: Path) -> Path:
    return img.parent.parent / 'labels' / f'{img.stem}.txt'


def _inst_path(img: Path) -> Path:
    return img.parent.parent / 'labels' / f'{img.stem}_inst.png'


def _poly_line(cid: int, pts: np.ndarray, h: int, w: int) -> Optional[str]:
    if len(pts) < 3:
        return None
    coords = []
    for x, y in pts:
        coords.append(f'{float(np.clip(x / w, 0, 1)):.6f}')
        coords.append(f'{float(np.clip(y / h, 0, 1)):.6f}')
    return f'{cid} ' + ' '.join(coords)


def mask_to_yolo(mask: np.ndarray, inst: Optional[np.ndarray] = None) -> List[str]:
    """Berry lines follow instance ids when inst is present; other classes stay CC."""
    h, w = mask.shape[:2]
    lines: List[str] = []
    berry_from_inst = inst is not None and int((inst > 0).sum()) > 0
    if berry_from_inst:
        for iid in np.unique(inst):
            iid = int(iid)
            if iid <= 0:
                continue
            m = ((inst == iid) & (mask == PIX_BERRY)).astype(np.uint8)
            if int(m.sum()) < 8:
                continue
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            c = max(contours, key=cv2.contourArea)
            approx = cv2.approxPolyDP(c, max(1.0, 0.010 * cv2.arcLength(c, True)), True)
            line = _poly_line(0, approx.reshape(-1, 2).astype(np.float64), h, w)
            if line:
                lines.append(line)
    for pix, cid in ((PIX_BERRY, 0), (PIX_BRANCH, 1), (PIX_RIGID, 2), (PIX_EGO, 3)):
        if cid == 0 and berry_from_inst:
            continue
        m = (mask == pix).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        simplify = 0.004 if cid == 1 else 0.010
        min_area = 8 if cid != 2 else 80
        for c in contours:
            if cv2.contourArea(c) < min_area:
                continue
            approx = cv2.approxPolyDP(c, max(1.0, simplify * cv2.arcLength(c, True)), True)
            line = _poly_line(cid, approx.reshape(-1, 2).astype(np.float64), h, w)
            if line:
                lines.append(line)
    return lines


class Annotator:
    def __init__(self, data: Path, queue: Optional[set[str] | list[str]] = None) -> None:
        self.paths = _list_images(data)
        if queue:
            stems = list(queue) if not isinstance(queue, set) else list(queue)
            order = {s: i for i, s in enumerate(stems)}
            self.paths = [p for p in self.paths if p.stem in order]
            self.paths.sort(key=lambda p: order.get(p.stem, 10**9))
        if not self.paths:
            raise SystemExit(f'no images under {data}/train|val/images')
        self.idx = 0
        self.cls = PIX_BERRY
        self.mode = 'berry'  # 默认点果；2/3/4 可跳过
        self.radius = 4
        self.use_sam = True
        self.zoom = 1.0
        self.mask: Optional[np.ndarray] = None
        self.inst: Optional[np.ndarray] = None
        self.bgr: Optional[np.ndarray] = None
        self._undo: List[Tuple[np.ndarray, np.ndarray]] = []
        self._sam = None
        self._yolo = None
        self._yolo_boxes: Optional[List[Tuple[int, int, int, int]]] = None
        self._paint = False
        self._erase_drag = False
        self._mx = 0
        self._my = 0
        self._view = (0, 0, 1.0)  # x0, y0, scale of displayed crop
        self._loupe_disp = None
        self._loupe_src = None
        self._load()

    def _push_undo(self) -> None:
        if self.mask is not None and self.inst is not None:
            self._undo.append((self.mask.copy(), self.inst.copy()))
            if len(self._undo) > 20:
                self._undo.pop(0)

    def _load_yolo(self):
        if self._yolo is not None:
            return self._yolo
        try:
            from ultralytics import YOLO
            if YOLO_WEIGHTS.is_file():
                self._yolo = YOLO(str(YOLO_WEIGHTS))
                print('[yolo] ready')
        except Exception as exc:
            print(f'[yolo] skip: {exc}')
            self._yolo = False
        return self._yolo

    def _load_sam(self):
        if self._sam is not None:
            return self._sam
        try:
            from ultralytics import SAM
            print('[sam] loading mobile_sam (first time may download) ...')
            self._sam = SAM('mobile_sam.pt')
            print('[sam] ready')
        except Exception as exc:
            print(f'[sam] unavailable: {exc}')
            self._sam = False
        return self._sam

    def _load(self) -> None:
        path = self.paths[self.idx]
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise SystemExit(f'bad image {path}')
        self.bgr = bgr
        mp = _mask_path(path)
        if mp.is_file():
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            self.mask = m if m is not None and m.shape[:2] == bgr.shape[:2] else np.zeros(bgr.shape[:2], np.uint8)
        else:
            self.mask = np.zeros(bgr.shape[:2], np.uint8)
        ip = _inst_path(path)
        self.inst = np.zeros(bgr.shape[:2], dtype=np.uint16)
        loaded_inst = False
        if ip.is_file():
            im = cv2.imread(str(ip), cv2.IMREAD_UNCHANGED)
            if im is not None and im.shape[:2] == bgr.shape[:2]:
                self.inst = im.astype(np.uint16)
                loaded_inst = True
        if not loaded_inst:
            txt = _txt_path(path)
            if txt.is_file():
                from picking_perception.instance_gt import maps_from_yolo_seg
                _sem, inst, _c = maps_from_yolo_seg(txt, bgr.shape[0], bgr.shape[1])
                self.inst = inst
        self._undo.clear()
        self._yolo_boxes = None
        self._apply_cam_defaults()
        n_berry_inst = len([i for i in np.unique(self.inst) if i > 0])
        kind = 'fixed-disk' if self._is_fixed() else 'wrist-sam'
        print(f'[{self.idx + 1}/{len(self.paths)}] {path.name}  '
              f'berry={(self.mask == 1).sum()} branch={(self.mask == 2).sum()} '
              f'rigid={(self.mask == 3).sum()} ego={(self.mask == 4).sum()} '
              f'fruits={n_berry_inst}  {kind} r={self.radius} zoom={self.zoom:g}')
        if self._is_fixed() and n_berry_inst:
            areas = [int((self.inst == i).sum()) for i in np.unique(self.inst) if i > 0]
            if areas and float(np.median(areas)) > 200:
                print('  本张圆偏大（上次一点整簇）。按 c 清空，再用小圆点果心。')

    def _save(self) -> None:
        path = self.paths[self.idx]
        lab = path.parent.parent / 'labels'
        lab.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(_mask_path(path)), self.mask)
        cv2.imwrite(str(_inst_path(path)), self.inst.astype(np.uint16))
        lines = mask_to_yolo(self.mask, self.inst)
        _txt_path(path).write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
        vis = self._render(hud=False)
        ov = ROOT / 'datasets' / 'scene_seg' / 'overlays'
        overlay_dir = ov
        overlay_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(overlay_dir / f'{path.stem}.jpg'), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        n_fruits = len([i for i in np.unique(self.inst) if i > 0])
        print(f'  saved {path.stem}  polys={len(lines)} fruits={n_fruits}')

    def _fill_yolo(self) -> None:
        print('[annotate] skip YOLO ellipses — use key 1 + click (SAM) so the mask hugs the fruit')

    def _ensure_yolo_boxes(self) -> List[Tuple[int, int, int, int]]:
        if self._yolo_boxes is not None:
            return self._yolo_boxes
        self._yolo_boxes = []
        model = self._load_yolo()
        if not model:
            return self._yolo_boxes
        try:
            rgb = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2RGB)
            res = model.predict(rgb, conf=0.22, verbose=False)[0]
        except Exception as exc:
            print(f'[yolo] predict skip: {exc}')
            return self._yolo_boxes
        names = res.names or {}
        if res.boxes is None:
            return self._yolo_boxes
        h, w = self.mask.shape
        max_area = 0.03 * h * w
        for box, cls in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy()):
            name = str(names.get(int(cls), '')).lower()
            if name in ('cluster', 'plant'):
                continue
            if int(cls) != 0 and name not in ('berry', 'blueberry'):
                continue
            x0, y0, x1, y1 = [int(v) for v in box]
            if (x1 - x0) * (y1 - y0) > max_area:
                continue
            self._yolo_boxes.append((x0, y0, x1, y1))
        return self._yolo_boxes

    def _is_fixed(self) -> bool:
        return '_fixed_' in self.paths[self.idx].stem

    def _apply_cam_defaults(self) -> None:
        if self._is_fixed():
            self.radius = 4
            self.use_sam = False
            self.zoom = 1.0
        else:
            self.radius = 10
            self.use_sam = True
            self.zoom = 1.0

    def _berry_radius(self) -> int:
        hi = 8 if self._is_fixed() else 24
        return max(2, min(hi, int(self.radius)))

    def _next_berry_id(self) -> int:
        return int(self.inst.max()) + 1 if self.inst is not None else 1

    def _stamp_region(self, region: np.ndarray) -> None:
        self.mask[region] = PIX_BERRY
        nid = self._next_berry_id()
        self.inst[region] = np.uint16(max(nid, 1))

    def _stamp_disk(self, x: int, y: int) -> None:
        h, w = self.mask.shape
        r = self._berry_radius()
        yy, xx = np.ogrid[:h, :w]
        disk = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
        self._stamp_region(disk)

    def _stamp_one_berry(self, x: int, y: int) -> None:
        """One click → one fruit. Fixed cam: small disk. Wrist: SAM clipped to radius."""
        h, w = self.mask.shape
        if not (0 <= x < w and 0 <= y < h):
            return
        if self.use_sam and self._load_sam():
            self._sam_at(x, y)
            return
        self._stamp_disk(x, y)

    def _color_in_disk(self, x: int, y: int, r: int) -> np.ndarray:
        h, w = self.mask.shape
        yy, xx = np.ogrid[:h, :w]
        disk = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
        hsv = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
        seed = hsv[y, x]
        dh = np.abs(hsv[:, :, 0] - int(seed[0]))
        dh = np.minimum(dh, 180 - dh)
        close = (dh < 18) & (np.abs(hsv[:, :, 1] - int(seed[1])) < 70) & (
            np.abs(hsv[:, :, 2] - int(seed[2])) < 70)
        inside = (close & disk).astype(np.uint8)
        _, lab = cv2.connectedComponents(inside, connectivity=8)
        lid = int(lab[y, x])
        if lid == 0:
            return np.zeros((h, w), dtype=bool)
        return lab == lid

    def _clip_to_berry(self, hit: np.ndarray, x: int, y: int) -> np.ndarray:
        r = self._berry_radius()
        yy, xx = np.ogrid[:hit.shape[0], :hit.shape[1]]
        disk = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
        local = (hit.astype(bool) & disk).astype(np.uint8)
        _, lab = cv2.connectedComponents(local, connectivity=8)
        lid = int(lab[y, x]) if 0 <= y < lab.shape[0] and 0 <= x < lab.shape[1] else 0
        if lid == 0:
            return disk
        clipped = lab == lid
        max_area = int(2.2 * np.pi * r * r)
        if int(clipped.sum()) > max_area:
            return disk
        return clipped

    def _sam_at(self, x: int, y: int) -> None:
        sam = self._load_sam()
        if not sam:
            r = max(6, self.radius)
            yy, xx = np.ogrid[:self.mask.shape[0], :self.mask.shape[1]]
            disk = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
            if self.cls == PIX_BERRY:
                self._stamp_region(disk)
            else:
                self.mask[disk] = np.uint8(self.cls)
            return
        rgb = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2RGB)
        try:
            res = sam.predict(rgb, points=[[x, y]], labels=[1], verbose=False)[0]
        except Exception as exc:
            print(f'sam fail: {exc}')
            return
        if res.masks is None or res.masks.data is None or len(res.masks.data) == 0:
            return
        m = res.masks.data[0].cpu().numpy()
        if m.shape != self.mask.shape:
            m = cv2.resize(m.astype(np.uint8), (self.mask.shape[1], self.mask.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        hit = m > 0.5
        if self.cls == PIX_BERRY:
            hit = self._clip_to_berry(hit, x, y)
            max_area = int(2.2 * np.pi * self._berry_radius() ** 2)
            if int(hit.sum()) > max_area:
                self._stamp_disk(x, y)
                return
        self._apply_hit(hit)

    def _flood(self, x: int, y: int) -> None:
        h, w = self.mask.shape
        hsv = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2HSV)
        seed = hsv[y, x].astype(np.int16)
        diff = np.abs(hsv.astype(np.int16) - seed)
        close = (diff[:, :, 0] < 12) & (diff[:, :, 1] < 50) & (diff[:, :, 2] < 50)
        # keep only the clicked connected component
        cc = close.astype(np.uint8)
        num, lab = cv2.connectedComponents(cc, connectivity=8)
        if not (0 <= y < h and 0 <= x < w):
            return
        lid = int(lab[y, x])
        if lid == 0:
            return
        hit = lab == lid
        if self.cls == PIX_BERRY:
            hit = self._clip_to_berry(hit, x, y)
        self._apply_hit(hit)

    def _apply_hit(self, hit: np.ndarray) -> None:
        if self.cls == 0:
            self.mask[hit] = 0
            self.inst[hit] = 0
        elif self.cls == PIX_BERRY:
            self._stamp_region(hit)
        else:
            self.mask[hit] = np.uint8(self.cls)
            self.inst[hit] = 0

    def _erase_radius(self) -> int:
        if self._is_fixed():
            return max(self._berry_radius() + 1, 4)
        return max(int(self.radius * 2), 14)

    def _erase_at(self, x: int, y: int) -> None:
        r = self._erase_radius()
        cv2.circle(self.mask, (x, y), r, 0, -1)
        cv2.circle(self.inst, (x, y), r, 0, -1)

    def _erase_blob(self, x: int, y: int) -> None:
        h, w = self.mask.shape
        if not (0 <= x < w and 0 <= y < h):
            return
        pix = int(self.mask[y, x])
        if pix == 0:
            self._erase_at(x, y)
            return
        if pix == PIX_BERRY and int(self.inst[y, x]) > 0:
            iid = int(self.inst[y, x])
            self.mask[self.inst == iid] = 0
            self.inst[self.inst == iid] = 0
            return
        num, lab = cv2.connectedComponents((self.mask == pix).astype(np.uint8), connectivity=8)
        lid = int(lab[y, x])
        if lid == 0:
            return
        self.mask[lab == lid] = 0
        self.inst[lab == lid] = 0

    def _disp_to_img(self, x: int, y: int) -> Tuple[int, int]:
        x0, y0, scale = self._view
        return int(x0 + x / scale), int(y0 + y / scale)

    def _nudge_radius(self, delta: int) -> None:
        hi = 8 if self._is_fixed() else 24
        self.radius = max(2, min(hi, int(self.radius) + delta))

    def _paste_loupe(self, disp: np.ndarray, vis: np.ndarray) -> np.ndarray:
        """Corner magnifier of the cursor. Main view stays the full image."""
        self._loupe_disp = None
        self._loupe_src = None
        h, w = vis.shape[:2]
        mx, my = int(self._mx), int(self._my)
        if not (0 <= mx < w and 0 <= my < h):
            return disp
        src = 28
        mag = 5
        x0 = max(0, mx - src)
        y0 = max(0, my - src)
        x1 = min(w, mx + src)
        y1 = min(h, my + src)
        patch = vis[y0:y1, x0:x1]
        if patch.size == 0:
            return disp
        side = src * 2 * mag
        big = cv2.resize(patch, (side, side), interpolation=cv2.INTER_NEAREST)
        sx = side / max(patch.shape[1], 1)
        sy = side / max(patch.shape[0], 1)
        cx = int((mx - x0) * sx)
        cy = int((my - y0) * sy)
        rr = max(2, int(self._berry_radius() * sx))
        cv2.circle(big, (cx, cy), rr, (0, 255, 255), 1)
        cv2.circle(big, (cx, cy), 2, (0, 255, 255), -1)
        cv2.rectangle(big, (0, 0), (big.shape[1] - 1, big.shape[0] - 1), (0, 255, 255), 1)
        pad = 8
        px = max(0, disp.shape[1] - big.shape[1] - pad)
        py = max(36, disp.shape[0] - big.shape[0] - pad)
        bh, bw = big.shape[:2]
        roi = disp[py:py + bh, px:px + bw]
        if roi.shape[0] != bh or roi.shape[1] != bw:
            return disp
        disp[py:py + bh, px:px + bw] = big
        self._loupe_disp = (px, py, bw, bh)
        self._loupe_src = (x0, y0, x1, y1)
        return disp

    def _on_mouse(self, event, x, y, flags, _param) -> None:
        try:
            if event == cv2.EVENT_MOUSEWHEEL:
                delta = 1 if (flags >> 16) > 0 else -1
                self._nudge_radius(delta)
                return
            if y < 36 and event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
                return
            if self._loupe_disp is not None and self._loupe_src is not None:
                px, py, bw, bh = self._loupe_disp
                if px <= x < px + bw and py <= y < py + bh:
                    if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN,
                                 cv2.EVENT_MBUTTONDOWN, cv2.EVENT_RBUTTONDBLCLK):
                        sx0, sy0, sx1, sy1 = self._loupe_src
                        ix = int(sx0 + (x - px) / max(bw, 1) * (sx1 - sx0))
                        iy = int(sy0 + (y - py) / max(bh, 1) * (sy1 - sy0))
                        self._on_mouse_inner(event, ix, iy, flags)
                    return
            ix, iy = self._disp_to_img(int(x), int(y))
            self._on_mouse_inner(event, ix, iy, flags)
        except Exception as exc:
            print(f'mouse: {exc}')

    def _on_mouse_inner(self, event, x, y, flags) -> None:
        if self.mask is None:
            return
        self._mx, self._my = int(x), int(y)
        rbtn = bool(flags & 2)       # EVENT_FLAG_RBUTTON
        shift = bool(flags & 16)    # EVENT_FLAG_SHIFTKEY
        ctrl = bool(flags & 8)      # EVENT_FLAG_CTRLKEY
        if event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN, cv2.EVENT_RBUTTONDBLCLK) or (
                event == cv2.EVENT_LBUTTONDOWN and (shift or ctrl)):
            self._push_undo()
            self._erase_drag = True
            self._erase_at(x, y)
            return
        if event == cv2.EVENT_MOUSEMOVE and (self._erase_drag or rbtn):
            self._erase_at(x, y)
            return
        if event in (cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP, cv2.EVENT_LBUTTONUP):
            self._paint = False
            self._erase_drag = False
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self._push_undo()
            if self.cls == 0:
                self._erase_drag = True
                self._erase_at(x, y)
            elif self.cls == PIX_BERRY:
                self._stamp_one_berry(x, y)
            elif self.mode == 'sam':
                self._sam_at(x, y)
            elif self.mode == 'flood':
                self._flood(x, y)
            else:
                self._paint = True
                cv2.circle(self.mask, (x, y), self.radius, int(self.cls), -1)
                cv2.circle(self.inst, (x, y), self.radius, 0, -1)
        elif event == cv2.EVENT_MOUSEMOVE and self._paint and self.mode == 'brush':
            cv2.circle(self.mask, (x, y), self.radius, int(self.cls), -1)
            cv2.circle(self.inst, (x, y), self.radius, 0, -1)

    def _render(self, hud: bool = True) -> np.ndarray:
        vis = self.bgr.copy()
        tint = np.zeros_like(vis)
        for pix, col in TINT.items():
            if pix == PIX_BERRY:
                continue
            tint[self.mask == pix] = col
        unassigned = (self.mask == PIX_BERRY) & (self.inst == 0)
        tint[unassigned] = (160, 60, 160)
        if self.inst is not None:
            for iid in np.unique(self.inst):
                iid = int(iid)
                if iid <= 0:
                    continue
                hue = int((iid * 41) % 180)
                col = cv2.cvtColor(np.uint8([[[hue, 210, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
                tint[self.inst == iid] = (int(col[0]), int(col[1]), int(col[2]))
        vis = cv2.addWeighted(vis, 0.62, tint, 0.38, 0)
        if not hud:
            return vis
        r = self._berry_radius()
        cv2.circle(vis, (self._mx, self._my), r, (0, 255, 255), 1)
        scale = float(self.zoom)
        if scale != 1.0:
            disp = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        else:
            disp = vis
        self._view = (0, 0, scale)
        disp = self._paste_loupe(disp, vis)
        n_inst = len([i for i in np.unique(self.inst) if i > 0]) if self.inst is not None else 0
        cam = '全局小圆' if self._is_fixed() else '腕部'
        sam = 'SAM开' if self.use_sam else 'SAM关'
        bar = (f'[{self.idx + 1}/{len(self.paths)}] {cam}  fruits={n_inst}  '
               f'r={r}  {sam}  全图{scale:g}x  '
               f'滚轮半径  z全图缩放  d下一张  c清空')
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(disp, bar, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        return disp

    def run(self) -> None:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, 1280, 720)
        cv2.setMouseCallback(WIN, self._on_mouse)
        print(__doc__)
        while True:
            cv2.imshow(WIN, self._render())
            k = cv2.waitKey(16) & 0xFF
            if k == 255:
                continue
            if k in (ord('q'), 27):
                self._save()
                break
            if k == ord('1'):
                self.cls = PIX_BERRY
                self.mode = 'berry'
            elif k == ord('2'):
                self.cls = PIX_BRANCH
                self.mode = 'brush'
            elif k == ord('3'):
                self.cls = PIX_RIGID
            elif k == ord('4'):
                self.cls = PIX_EGO
            elif k in (ord('0'), ord('e')):
                self.cls = 0
                self.mode = 'brush'
            elif k == ord('x'):
                self._push_undo()
                self._erase_blob(self._mx, self._my)
            elif k == ord('b'):
                self.mode = 'brush'
            elif k == ord('s'):
                self.use_sam = not self.use_sam
                print(f'  SAM {"on" if self.use_sam else "off"}')
            elif k == ord('z'):
                order = (1.0, 1.5, 2.0)
                try:
                    i = order.index(float(self.zoom))
                    self.zoom = order[(i + 1) % len(order)]
                except ValueError:
                    self.zoom = 1.0
            elif k in (ord('-'), ord('[')):
                self._nudge_radius(-1)
            elif k in (ord('='), ord(']'), ord('+')):
                self._nudge_radius(1)
            elif k == ord('u') and self._undo:
                self.mask, self.inst = self._undo.pop()
            elif k == ord('c'):
                self._push_undo()
                self.mask[:] = 0
                self.inst[:] = 0
            elif k in (ord('d'), 13):
                self._save()
                self.idx = min(len(self.paths) - 1, self.idx + 1)
                self._load()
            elif k == ord('a'):
                self._save()
                self.idx = max(0, self.idx - 1)
                self._load()
        cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=str(ROOT / 'datasets' / 'scene_seg'))
    parser.add_argument('--start', default='', help='filename substring to start from')
    parser.add_argument('--queue', default='', help='txt of image stems (unlabeled queue)')
    args = parser.parse_args()
    queue: Optional[list[str]] = None
    if args.queue:
        stems = []
        for line in Path(args.queue).read_text(encoding='utf-8').splitlines():
            s = line.strip()
            if s and not s.startswith('#'):
                stems.append(Path(s).stem)
        queue = stems
    ann = Annotator(Path(args.data), queue=queue)
    if args.start:
        for i, p in enumerate(ann.paths):
            if args.start in p.name:
                ann.idx = i
                ann._load()
                break
    ann.run()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
