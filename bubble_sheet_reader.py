"""
Robust bubble-sheet reader for phone photos of printed, hand-filled MCQ sheets.

Pipeline
--------
1. find_markers   : ArUco detection with fall-backs (contrast boost, upscale, faded-marker blob search).
2. warp           : perspective-correct the sheet using the marker CENTRES, into a canonical frame
                    whose aspect ratio is the real one of the printed sheet (see Layout).
3. flat-field     : divide by the local paper brightness -> shadows / uneven light disappear.
4. locate_bubbles : find the printed circles (Hough), then fit the known grid layout to them
                    (translation vote + affine refinement).  No contour/aspect-ratio filtering, so
                    scribbled, half-filled or crossed-out bubbles cannot break the row/column order.
5. score_bubbles  : ink coverage inside each bubble (colour-agnostic: blue/black/red pen, pencil).
6. decide         : per-sheet adaptive threshold + per-question status
                    (ok / blank / multiple / faint) so doubtful answers are flagged, not guessed.
7. read_student_id: optional TrOCR on a cleaned crop located from the layout (crop can be saved for review).

Usage
-----
    from bubble_sheet_reader import BubbleSheetReader
    res = BubbleSheetReader().read("sheet.jpg", debug_dir="debug")
    print(res.answers)          # ['B', 'D', 'C', ..., None]
    print(res.needs_review)     # question numbers a human should look at
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ArUco ids used by the sheet generator
ID_TL, ID_TR, ID_BL = 30, 10, 34
ID_BR_5_OPTIONS, ID_BR_4_OPTIONS = 48, 49


# --------------------------------------------------------------------------------------
# Layout (all numbers in "canonical px": marker-centre rectangle is sheet_w x sheet_h,
# origin = centre of the top-left marker).  Measured from the printed sheets; if you change
# bubbleSheetGenerator.html, change these numbers (or run Layout calibration, see README).
# --------------------------------------------------------------------------------------
@dataclass
class Layout:
    n_questions: int = 20
    n_options: int = 4
    n_cols: int = 3                 # question columns on the sheet (questions run row by row)
    sheet_w: float = 640.0          # distance TL<->TR marker centres
    sheet_h: float = 1100.0         # distance TL<->BL marker centres
    first_x: float = 68.0           # x of option A of the first question column
    col_pitch: float = 202.0        # distance between question columns
    opt_pitch: float = 31.6         # distance between options inside a question
    first_y: float = 116.0          # y of the first row
    row_pitch: float = 40.0
    bubble_r: float = 13.0
    pad: float = 70.0               # extra margin kept around the marker rectangle when warping
    # student-id box (x0, y0, x1, y1) in the same units
    id_box: Tuple[float, float, float, float] = (484.0, 12.0, 626.0, 56.0)
    letters: str = "ABCDE"

    def template(self) -> np.ndarray:
        """(n_questions, n_options, 2) bubble centres (x, y) in canonical px."""
        t = np.zeros((self.n_questions, self.n_options, 2), np.float32)
        for q in range(self.n_questions):
            r, c = divmod(q, self.n_cols)
            for k in range(self.n_options):
                t[q, k] = (self.first_x + c * self.col_pitch + k * self.opt_pitch,
                           self.first_y + r * self.row_pitch)
        return t


@dataclass
class QuestionResult:
    number: int
    answer: Optional[str]
    status: str                      # ok | stray | faint | multiple | blank  (anything but ok -> review)
    fills: List[float]
    marked: List[str]


@dataclass
class SheetResult:
    answers: List[Optional[str]]
    questions: List[QuestionResult]
    student_id: Optional[str]
    warnings: List[str] = field(default_factory=list)
    threshold: float = 0.0
    n_options: int = 4
    matrix: Optional[np.ndarray] = None   # (n_questions, n_options) 0/1, drop-in for the old answer_matrix

    @property
    def needs_review(self) -> List[int]:
        return [q.number for q in self.questions if q.status != "ok"]


class SheetError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------
class BubbleSheetReader:
    def __init__(self, layout: Optional[Layout] = None, read_id: bool = True,
                 multiple_policy: str = "none"):
        """
        multiple_policy: what to put in `answers` when >1 bubble is marked
            "none"    -> None (flag for review)           [default, safest]
            "darkest" -> the darkest bubble (flag stays)
        """
        self.layout = layout or Layout()
        self.read_id = read_id
        self.multiple_policy = multiple_policy
        self._ocr = None

    # ------------------------------------------------------------------ public
    def read(self, image_path: str, debug_dir: Optional[str] = None) -> SheetResult:
        img = cv2.imread(image_path)                      # honours EXIF rotation
        if img is None:
            raise SheetError(f"cannot read {image_path}")
        return self.read_array(img, debug_dir, name=os.path.splitext(os.path.basename(image_path))[0])

    def read_array(self, img: np.ndarray, debug_dir: Optional[str] = None, name: str = "sheet") -> SheetResult:
        warnings: List[str] = []
        lay = self.layout
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)

        markers, nopt, w = self.find_markers(img)
        warnings += w
        if nopt and nopt != lay.n_options:
            lay = Layout(**{**lay.__dict__, "n_options": nopt})
            warnings.append(f"layout switched to {nopt} options (from marker id)")

        warped, scale, H = self.warp(img, markers, lay)
        gray = self.flat_field(warped, scale * lay.bubble_r)
        centres, n_matched, w = self.locate_bubbles(gray, lay, scale)
        warnings += w
        fills = self.score_bubbles(gray, centres, scale * lay.bubble_r)
        res = self.decide(fills, lay)
        res.warnings = warnings
        if n_matched < 30:
            res.warnings.append(f"only {n_matched} printed circles matched the grid - check the debug image")

        if self.read_id:
            try:
                res.student_id = self.read_student_id(warped, scale, lay, debug_dir, name)
            except Exception as e:                       # ID reading must never kill the answers
                res.warnings.append(f"student id not read: {e}")

        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, f"{name}_overlay.jpg"),
                        self.draw_overlay(warped, centres, scale * lay.bubble_r, res))
        return res

    # ------------------------------------------------------------------ 1. markers
    @staticmethod
    def _detect_markers(gray: np.ndarray) -> Dict[int, np.ndarray]:
        dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        params = cv2.aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin, params.adaptiveThreshWinSizeMax = 3, 53
        params.adaptiveThreshWinSizeStep = 4
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        det = cv2.aruco.ArucoDetector(dic, params)
        found: Dict[int, np.ndarray] = {}
        wanted = {ID_TL, ID_TR, ID_BL, ID_BR_4_OPTIONS, ID_BR_5_OPTIONS}
        clahe = cv2.createCLAHE(3.0, (8, 8))
        passes = [(gray, 1.0), (clahe.apply(gray), 1.0),
                  (cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), 2.0)]
        for g, s in passes:
            corners, ids, _ = det.detectMarkers(g)
            if ids is None:
                continue
            for i, c in zip(ids.flatten(), corners):
                if int(i) in wanted and int(i) not in found:
                    found[int(i)] = c[0] / s
            if {ID_TL, ID_TR, ID_BL} <= found.keys() and (ID_BR_4_OPTIONS in found or ID_BR_5_OPTIONS in found):
                break
        return found

    @staticmethod
    def _find_faded_marker(gray: np.ndarray, pred: np.ndarray, side: float) -> Optional[np.ndarray]:
        """Last resort for a marker that is too washed-out to decode (thin paper, glare, bad print):
        look for a dark, roughly square blob of the right size near the predicted position."""
        r = int(0.12 * max(gray.shape))
        x0, y0 = np.maximum(pred - r, 0).astype(int)
        x1, y1 = np.minimum(pred + r, [gray.shape[1], gray.shape[0]]).astype(int)
        crop = gray[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        bg = cv2.medianBlur(crop, 51).astype(np.float32)
        mask = ((crop / np.maximum(bg, 1)) < 0.88).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, _, st, cen = cv2.connectedComponentsWithStats(mask)
        best, best_d = None, 1e9
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if not (0.65 * side < w < 1.4 * side and 0.65 * side < h < 1.4 * side):
                continue
            if a / float(w * h) < 0.55:
                continue
            c = cen[i] + [x0, y0]
            d = np.linalg.norm(c - pred)
            if d < best_d:
                best, best_d = c, d
        return None if best is None else np.float32(best)

    def find_markers(self, img: np.ndarray):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found = self._detect_markers(gray)
        warnings: List[str] = []
        br_id = ID_BR_4_OPTIONS if ID_BR_4_OPTIONS in found else ID_BR_5_OPTIONS if ID_BR_5_OPTIONS in found else None
        nopt = None if br_id is None else (4 if br_id == ID_BR_4_OPTIONS else 5)
        centres = {i: found[i].mean(0) for i in found}
        sides = [np.linalg.norm(found[i][1] - found[i][0]) for i in found]
        side = float(np.median(sides)) if sides else 25.0

        roles = {"TL": ID_TL, "TR": ID_TR, "BL": ID_BL, "BR": br_id}
        missing = [k for k, v in roles.items() if v is None or v not in found]
        if len(missing) == 1 and len(found) >= 3:
            role = missing[0]
            c = {k: centres[v] for k, v in roles.items() if v in centres}
            pred = {"TL": lambda: c["TR"] + c["BL"] - c["BR"],
                    "TR": lambda: c["TL"] + c["BR"] - c["BL"],
                    "BL": lambda: c["TL"] + c["BR"] - c["TR"],
                    "BR": lambda: c["TR"] + c["BL"] - c["TL"]}[role]()
            p = self._find_faded_marker(gray, np.float32(pred), side)
            if p is None:
                raise SheetError(f"marker {role} not found (faded/occluded). Re-take the photo.")
            warnings.append(f"marker {role} was too faint to decode; located by shape")
            centres_role = p
            ids_for_role = roles[role] if roles[role] is not None else ID_BR_4_OPTIONS
            centres[ids_for_role] = centres_role
            roles[role] = ids_for_role
        elif missing:
            raise SheetError(f"markers missing: {missing}")

        pts = np.float32([centres[roles[k]] for k in ("TL", "TR", "BR", "BL")])
        return pts, nopt, warnings

    # ------------------------------------------------------------------ 2. warp
    @staticmethod
    def warp(img: np.ndarray, pts: np.ndarray, lay: Layout):
        top = np.linalg.norm(pts[1] - pts[0])
        bot = np.linalg.norm(pts[2] - pts[3])
        scale = float(np.clip(0.5 * (top + bot) / lay.sheet_w, 1.0, 2.0))   # keep source resolution
        W, H = lay.sheet_w * scale, lay.sheet_h * scale
        P = lay.pad * scale
        dst = np.float32([[P, P], [P + W, P], [P + W, P + H], [P, P + H]])
        M = cv2.getPerspectiveTransform(pts, dst)
        out = cv2.warpPerspective(img, M, (int(W + 2 * P), int(H + 2 * P)), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        return out, scale, M

    # ------------------------------------------------------------------ 3. flat-field
    @staticmethod
    def flat_field(warped: np.ndarray, r: float) -> np.ndarray:
        """Return a float image where clean paper ~1.0 and ink is well below, regardless of shadows."""
        g = warped.min(axis=2)                       # min over B,G,R: saturated inks (blue/red) become very dark
        k = int(2.6 * r) | 1
        bg = cv2.dilate(g, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        bg = cv2.GaussianBlur(bg, (0, 0), r)
        return np.clip(g.astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0), 0, 1.2)

    # ------------------------------------------------------------------ 4. locate bubbles
    def locate_bubbles(self, norm: np.ndarray, lay: Layout, scale: float):
        warnings: List[str] = []
        P = lay.pad * scale
        R = lay.bubble_r * scale
        tmpl = lay.template().reshape(-1, 2) * scale + P

        u8 = np.clip(norm * 235, 0, 255).astype(np.uint8)
        u8 = cv2.GaussianBlur(u8, (3, 3), 0)
        # only look at the answer area (with generous margin) to avoid hits on text
        x0, y0 = tmpl.min(0) - 5 * R
        x1, y1 = tmpl.max(0) + 5 * R
        x0, y0 = max(int(x0), 0), max(int(y0), 0)
        roi = u8[y0:int(y1), x0:int(x1)]
        circ = cv2.HoughCircles(roi, cv2.HOUGH_GRADIENT, dp=1, minDist=1.4 * R,
                                param1=100, param2=16, minRadius=int(0.75 * R), maxRadius=int(1.3 * R))
        det = np.zeros((0, 2), np.float32) if circ is None else circ[0][:, :2] + [x0, y0]

        if len(det) < 12:
            warnings.append("few circles detected; using nominal grid position")
            return tmpl.reshape(lay.n_questions, lay.n_options, 2), 0, warnings

        # (a) translation vote: which shift makes most template points land on a detected circle?
        tol = 0.35 * lay.opt_pitch * scale
        best, best_t = -1, np.zeros(2, np.float32)
        rng = int(0.6 * lay.row_pitch * scale)
        step = max(1, int(scale))
        for dy in range(-rng, rng + 1, step):
            for dx in range(-rng, rng + 1, step):
                d = np.linalg.norm((tmpl + [dx, dy])[:, None, :] - det[None, :, :], axis=2).min(1)
                score = (d < tol).sum() - 0.02 * d.clip(max=tol).sum() / tol
                if score > best:
                    best, best_t = score, np.float32([dx, dy])
        M = np.hstack([np.eye(2, dtype=np.float32), best_t[:, None]])

        # (b) a few rounds of robust affine refinement (copes with residual perspective / paper bend)
        n_matched = 0
        for gate in (tol, 0.8 * tol, 0.6 * tol):
            cur = tmpl @ M[:, :2].T + M[:, 2]
            dist = np.linalg.norm(cur[:, None, :] - det[None, :, :], axis=2)
            j = dist.argmin(1)
            ok = dist[np.arange(len(cur)), j] < gate
            n_matched = int(ok.sum())
            if n_matched < 12:
                break
            A, _ = cv2.estimateAffine2D(tmpl[ok], det[j[ok]], method=cv2.LMEDS)
            if A is None:
                break
            # keep the transform close to a similarity: reject crazy shears/scales
            sx, sy = np.linalg.norm(A[:, 0]), np.linalg.norm(A[:, 1])
            if not (0.9 < sx < 1.1 and 0.9 < sy < 1.1):
                warnings.append("affine refinement rejected (implausible scale)")
                break
            M = A.astype(np.float32)
        centres = tmpl @ M[:, :2].T + M[:, 2]
        return centres.reshape(lay.n_questions, lay.n_options, 2), n_matched, warnings

    # ------------------------------------------------------------------ 5. score
    @staticmethod
    def score_bubbles(norm: np.ndarray, centres: np.ndarray, r: float) -> np.ndarray:
        """Return fills[q, k] in 0..1: fraction of 'inked' pixels in the inner disk of each bubble.
        The disk is 0.72 r so the printed ring (and small mis-alignments) are not counted."""
        h, w = norm.shape
        fills = np.zeros(centres.shape[:2], np.float32)
        rin = 0.72 * r
        yy, xx = np.mgrid[-int(r) - 2:int(r) + 3, -int(r) - 2:int(r) + 3]
        disk = (xx ** 2 + yy ** 2) <= rin ** 2
        # ink = clearly darker than the local paper. Printed letters (A-D) cover ~10-15% only.
        ink = norm < 0.78
        for q in range(centres.shape[0]):
            for k in range(centres.shape[1]):
                cx, cy = centres[q, k]
                ix, iy = int(round(cx)), int(round(cy))
                ys, xs = iy + yy, ix + xx
                m = disk & (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
                fills[q, k] = ink[ys[m], xs[m]].mean() if m.any() else 0.0
        return fills

    # ------------------------------------------------------------------ 6. decide
    def decide(self, fills: np.ndarray, lay: Layout) -> SheetResult:
        flat = fills.ravel()
        # sheet-adaptive threshold between "empty (letters only)" and "marked" via Otsu on the fills
        u8 = np.clip(flat * 255, 0, 255).astype(np.uint8).reshape(-1, 1)
        otsu, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # unmarked bubbles sit at ~0.15-0.30 (printed letter + ring noise), marked ones at ~0.85-1.0
        thr = float(np.clip(otsu / 255.0, 0.45, 0.65))
        faint_thr = 0.35

        qs, answers, mat = [], [], np.zeros_like(fills, dtype=np.uint8)
        for q in range(fills.shape[0]):
            f = fills[q]
            top = float(f.max())
            # a bubble counts as marked if it is dark in absolute terms AND comparable to the darkest one
            # (a stray pen touch / eraser smudge next to a solid answer is not a second answer)
            marked = [k for k in range(len(f)) if f[k] >= thr and f[k] >= 0.7 * top]
            stray = [k for k in range(len(f)) if f[k] >= 0.35 and k not in marked and top >= thr]
            if len(marked) == 1:
                status, ans = ("stray", lay.letters[marked[0]]) if stray else ("ok", lay.letters[marked[0]])
            elif len(marked) > 1:
                status = "multiple"
                ans = lay.letters[int(f.argmax())] if self.multiple_policy == "darkest" else None
            else:
                k = int(f.argmax())
                if f[k] >= faint_thr and (np.sort(f)[-2] < 0.6 * f[k]):
                    status, ans, marked = "faint", lay.letters[k], [k]     # light pencil etc: answer but flag
                else:
                    status, ans = "blank", None
            for k in marked:
                mat[q, k] = 1
            answers.append(ans)
            qs.append(QuestionResult(q + 1, ans, status, [round(float(x), 2) for x in f],
                                     [lay.letters[k] for k in marked]))
        return SheetResult(answers, qs, None, threshold=thr, n_options=lay.n_options, matrix=mat)

    # ------------------------------------------------------------------ 7. student id
    def read_student_id(self, warped, scale, lay: Layout, debug_dir=None, name="sheet") -> Optional[str]:
        """Crop the ID box from the layout (not from 'largest contour'), remove the printed frame,
        the paper's show-through and shadows, then OCR. The cleaned crop is saved for human review."""
        P = lay.pad * scale
        bx0, by0, bx1, by1 = [v * scale + P for v in lay.id_box]
        box = self._find_id_frame(warped, (bx0, by0, bx1, by1), scale)      # follows the real printed frame
        x0, y0, x1, y1 = [int(v) for v in box]
        crop = warped[max(y0, 0):y1, max(x0, 0):x1]
        g = crop.min(axis=2)
        g = cv2.resize(g, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        k = int(0.6 * g.shape[0]) | 1
        bg = cv2.GaussianBlur(cv2.dilate(g, np.ones((k, k), np.uint8)), (0, 0), k / 3)
        n = g.astype(np.float32) / np.maximum(bg, 1)
        ink = (n < 0.62).astype(np.uint8) * 255            # pen strokes; back-page show-through is lighter
        h, w = ink.shape
        # drop anything that touches the crop border (the printed frame) and small specks
        cnt, lab, st, _ = cv2.connectedComponentsWithStats(ink)
        keep = np.zeros_like(ink)
        for i in range(1, cnt):
            x, y, cw, ch, area = st[i]
            touches = x <= 2 or y <= 2 or x + cw >= w - 2 or y + ch >= h - 2
            if area < 0.004 * h * w or (touches and (cw > 0.5 * w or ch > 0.7 * h) and area < 0.25 * cw * ch * 4):
                continue
            if ch < 0.10 * h and cw < 0.10 * h:           # specks / dots
                continue
            if cw > 3 * ch and ch < 0.25 * h:             # leftover bits of the printed frame
                continue
            keep[lab == i] = 255
        ink = cv2.dilate(keep, np.ones((3, 3), np.uint8))
        ys, xs = np.nonzero(ink)
        if len(xs) < 40:
            return None
        ink = ink[max(ys.min() - 10, 0):ys.max() + 10, max(xs.min() - 10, 0):xs.max() + 10]
        img = cv2.copyMakeBorder(255 - ink, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=255)
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, f"{name}_id.png"), img)
        digits = re.sub(r"\D", "", self._ocr_digits(img))
        return digits or None

    @staticmethod
    def _find_id_frame(warped, nominal, scale):
        """Find the printed rectangle of the ID field near its nominal position (+-20 px).
        Falls back to the nominal box if no rectangle-like component is found."""
        x0, y0, x1, y1 = nominal
        m = 22 * scale
        X0, Y0 = int(max(x0 - m, 0)), int(max(y0 - m, 0))
        X1, Y1 = int(x1 + m), int(y1 + m)
        g = warped[Y0:Y1, X0:X1].min(axis=2)
        bg = cv2.dilate(g, np.ones((25, 25), np.uint8))
        n = g.astype(np.float32) / np.maximum(bg, 1)
        ink = ((n < 0.80).astype(np.uint8)) * 255
        ink = cv2.dilate(ink, np.ones((3, 3), np.uint8))
        cnt, _, st, _ = cv2.connectedComponentsWithStats(ink)
        bw, bh = x1 - x0, y1 - y0
        best, best_s = None, 0
        for i in range(1, cnt):
            x, y, w, h, a = st[i]
            if 0.75 * bw < w < 1.25 * bw and 0.5 * bh < h < 1.5 * bh and a > 0.4 * (w + h) * 2:
                sc = w * h
                if sc > best_s:
                    best, best_s = (x + X0, y + Y0, x + w + X0, y + h + Y0), sc
        if best is None:
            return nominal
        bx0, by0, bx1, by1 = best
        i = 3 * scale                                  # step inside the printed line
        return (bx0 + i, by0 + i, bx1 - i, by1 - i)

    def _ocr_digits(self, gray_crop: np.ndarray) -> str:
        if self._ocr is None:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch
            proc = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").eval()
            self._ocr = (proc, model, torch)
        proc, model, torch = self._ocr
        from PIL import Image
        pil = Image.fromarray(cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2RGB))
        pix = proc(images=pil, return_tensors="pt").pixel_values
        with torch.no_grad():
            ids = model.generate(pix, max_length=12, num_beams=5, early_stopping=True)
        return proc.batch_decode(ids, skip_special_tokens=True)[0]

    # ------------------------------------------------------------------ debug overlay
    @staticmethod
    def draw_overlay(warped, centres, r, res: SheetResult):
        out = warped.copy()
        col = {"ok": (0, 170, 0), "stray": (0, 170, 0), "faint": (0, 165, 255), "multiple": (0, 0, 255), "blank": (0, 0, 255)}
        for q, qr in enumerate(res.questions):
            for k in range(centres.shape[1]):
                c = tuple(int(round(v)) for v in centres[q, k])
                chosen = "ABCDE"[k] in qr.marked
                cv2.circle(out, c, int(r), col[qr.status] if chosen else (200, 200, 200), 2 if chosen else 1)
            if qr.status in ("blank", "multiple"):
                c = tuple(int(v) for v in centres[q, 0])
                cv2.putText(out, qr.status[:5], (c[0] - 55, c[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        return out


# --------------------------------------------------------------------------------------
# Backwards-compatible wrapper: same constructor / run_pipeline() / attributes as the old class.
# --------------------------------------------------------------------------------------
class BubbleSheetProcessor:
    """Drop-in replacement for the old BubbleSheetProcessor.

        p = BubbleSheetProcessor(image_path)
        student_id, answer_matrix = p.run_pipeline()

    answer_matrix is a (n_questions, nc) 0/1 numpy array like before (a row with no mark is all zeros;
    a row with several marks has several 1s, so the old grading code keeps working).
    Extra, optional: p.result (SheetResult) with .answers, .needs_review, .warnings, .questions.
    """

    def __init__(self, image_path, read_id: bool = True, debug_dir: Optional[str] = None):
        self.image_path = image_path
        self.read_id = read_id
        self.debug_dir = debug_dir
        self.nc = None
        self.student_id = None
        self.answer_matrix = None
        self.result: Optional[SheetResult] = None

    def run_pipeline(self):
        self.result = BubbleSheetReader(read_id=self.read_id).read(self.image_path, self.debug_dir)
        self.nc = self.result.n_options
        self.student_id = self.result.student_id or ""
        self.answer_matrix = self.result.matrix.astype(int)   # int64 like the old class (uint8 would wrap on subtraction)
        return self.student_id, self.answer_matrix


# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--debug", default="debug")
    ap.add_argument("--no-id", action="store_true")
    a = ap.parse_args()
    reader = BubbleSheetReader(read_id=not a.no_id)
    for p in a.images:
        try:
            r = reader.read(p, a.debug)
            print(f"{os.path.basename(p)}  id={r.student_id}  thr={r.threshold:.2f}")
            print("  answers:", "".join(x or "-" for x in r.answers))
            if r.needs_review:
                print("  REVIEW :", r.needs_review)
            for w in r.warnings:
                print("  warn   :", w)
        except SheetError as e:
            print(f"{os.path.basename(p)}  FAILED: {e}")
