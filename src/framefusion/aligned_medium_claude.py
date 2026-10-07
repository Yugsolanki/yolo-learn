#!/usr/bin/env python3
"""
Temporal fusion with an aligned median.

For each window of N frames:
  1. Pick a reference frame.
  2. Align every other frame to the reference with pyramid ECC
     (Enhanced Correlation Coefficient). If ECC fails, an ORB + RANSAC
     estimate gives ECC a better start.
  3. Drop frames that do not align (ECC correlation below --min-cc).
  4. Take the per-pixel median of the aligned frames. Pixels that a warp
     moved outside the frame do not vote.

The median removes impulse noise, compression blocks, and short occluders.
It also removes a clock second hand if the hand covers a pixel in fewer
than half of the frames. Keep the window short relative to the minute
hand: the minute hand moves 0.1 degree per second.

Input: a directory of images (sorted by file name) or a video file.
Output: one lossless PNG per window.

Requirements: opencv-python (or opencv-python-headless) >= 4.2, numpy.

Examples:
  python fuse_aligned_median.py captures/ out/ --window 9
  python fuse_aligned_median.py clip.mp4 out/ --window 15 --stride 15 \
      --roi 820,310,400,400 --motion homography
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

MOTIONS = {
    "translation": cv2.MOTION_TRANSLATION,
    "euclidean": cv2.MOTION_EUCLIDEAN,
    "affine": cv2.MOTION_AFFINE,
    "homography": cv2.MOTION_HOMOGRAPHY,
}


# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------

def iter_frames(src: Path, roi: tuple[int, int, int, int] | None) -> Iterator[tuple[str, np.ndarray]]:
    """Yield (name, BGR frame) from a directory of images or a video file."""
    def crop(img: np.ndarray) -> np.ndarray:
        if roi is None:
            return img
        x, y, w, h = roi
        return img[y:y + h, x:x + w]

    if src.is_dir():
        files = sorted(f for f in src.iterdir() if f.suffix.lower() in IMG_EXTS)
        for f in files:
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if img is None:
                print(f"[warn] cannot read {f}, skipped", file=sys.stderr)
                continue
            yield f.stem, crop(img)
    else:
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise SystemExit(f"cannot open video: {src}")
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield f"{src.stem}_{idx:06d}", crop(frame)
            idx += 1
        cap.release()


def iter_windows(frames: Iterator, n: int, stride: int) -> Iterator[list]:
    """Group frames into windows of n. stride < n gives overlapping windows."""
    buf: list = []
    skip = 0
    emitted = 0
    for item in frames:
        if skip:
            skip -= 1
            continue
        buf.append(item)
        if len(buf) == n:
            yield buf
            emitted += 1
            if stride >= n:
                buf, skip = [], stride - n
            else:
                buf = buf[stride:]
    # A clip shorter than one window still gives one output.
    if emitted == 0 and len(buf) >= 2:
        yield buf


# --------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------

def to_gray_f32(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return g.astype(np.float32) / 255.0


def identity_warp(motion: int) -> np.ndarray:
    if motion == cv2.MOTION_HOMOGRAPHY:
        return np.eye(3, dtype=np.float32)
    return np.eye(2, 3, dtype=np.float32)


def rescale_warp(warp: np.ndarray, factor: float, motion: int) -> np.ndarray:
    """Convert a warp to the coordinates of an image scaled by `factor`."""
    if motion == cv2.MOTION_HOMOGRAPHY:
        s = np.diag([factor, factor, 1.0]).astype(np.float32)
        return (s @ warp @ np.linalg.inv(s)).astype(np.float32)
    out = warp.copy()
    out[:, 2] *= factor
    return out


def pyramid_levels(shape: tuple[int, ...], requested: int, min_side: int = 64) -> int:
    """Limit the pyramid so that the coarsest level keeps min_side pixels."""
    side = min(shape[:2])
    levels = 1
    while levels < requested and side / (2 ** levels) >= min_side:
        levels += 1
    return levels


def ecc_align(ref_g: np.ndarray, mov_g: np.ndarray, motion: int, levels: int,
              iters: int, eps: float, init: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """Coarse-to-fine ECC. The warp maps reference coords to moving-frame coords."""
    refs, movs = [ref_g], [mov_g]
    for _ in range(levels - 1):
        refs.append(cv2.pyrDown(refs[-1]))
        movs.append(cv2.pyrDown(movs[-1]))

    warp = identity_warp(motion) if init is None else init.astype(np.float32)
    warp = rescale_warp(warp, 1.0 / 2 ** (levels - 1), motion)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)

    cc = -1.0
    for lvl in range(levels - 1, -1, -1):
        cc, warp = cv2.findTransformECC(refs[lvl], movs[lvl], warp, motion, criteria, None, 5)
        if lvl > 0:
            warp = rescale_warp(warp, 2.0, motion)
    return float(cc), warp


def orb_init(ref_u8: np.ndarray, mov_u8: np.ndarray, motion: int) -> np.ndarray | None:
    """Feature-based first guess for large PTZ offsets where ECC does not converge."""
    orb = cv2.ORB_create(4000)
    k1, d1 = orb.detectAndCompute(ref_u8, None)
    k2, d2 = orb.detectAndCompute(mov_u8, None)
    if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    matches = sorted(matches, key=lambda m: m.distance)[:500]
    if len(matches) < 10:
        return None
    p_ref = np.float32([k1[m.queryIdx].pt for m in matches])
    p_mov = np.float32([k2[m.trainIdx].pt for m in matches])

    if motion == cv2.MOTION_HOMOGRAPHY:
        warp, inl = cv2.findHomography(p_ref, p_mov, cv2.RANSAC, 3.0)
    elif motion == cv2.MOTION_AFFINE:
        warp, inl = cv2.estimateAffine2D(p_ref, p_mov, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    else:
        warp, inl = cv2.estimateAffinePartial2D(p_ref, p_mov, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if warp is None or inl is None or int(inl.sum()) < 10:
        return None
    warp = warp.astype(np.float32)

    if motion == cv2.MOTION_TRANSLATION:
        return np.array([[1, 0, warp[0, 2]], [0, 1, warp[1, 2]]], np.float32)
    if motion == cv2.MOTION_EUCLIDEAN:
        # ECC Euclidean has no scale term, so keep only the rotation.
        theta = np.arctan2(warp[1, 0], warp[0, 0])
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, warp[0, 2]], [s, c, warp[1, 2]]], np.float32)
    return warp


def align_to_ref(ref: np.ndarray, mov: np.ndarray, motion: int, levels: int,
                 iters: int, eps: float, min_cc: float):
    """Return (aligned frame, valid mask, cc). aligned is None if alignment fails."""
    h, w = ref.shape[:2]
    if mov.shape[:2] != (h, w):
        mov = cv2.resize(mov, (w, h), interpolation=cv2.INTER_AREA)

    ref_g, mov_g = to_gray_f32(ref), to_gray_f32(mov)
    cc, warp = -1.0, None
    try:
        cc, warp = ecc_align(ref_g, mov_g, motion, levels, iters, eps)
    except cv2.error:
        pass

    if warp is None or cc < min_cc:
        init = orb_init((ref_g * 255).astype(np.uint8), (mov_g * 255).astype(np.uint8), motion)
        if init is not None:
            try:
                cc2, warp2 = ecc_align(ref_g, mov_g, motion, levels, iters, eps, init=init)
                if cc2 > cc:
                    cc, warp = cc2, warp2
            except cv2.error:
                pass

    if warp is None or cc < min_cc:
        return None, None, cc

    warp_fn = cv2.warpPerspective if motion == cv2.MOTION_HOMOGRAPHY else cv2.warpAffine
    aligned = warp_fn(mov, warp, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = warp_fn(np.full((h, w), 255, np.uint8), warp, (w, h),
                    flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # Erode by one pixel so that interpolated border pixels do not vote.
    valid = cv2.erode(valid, np.ones((3, 3), np.uint8)) > 0
    return aligned, valid, cc


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------

def sharpness(img: np.ndarray) -> float:
    """Laplacian variance after a blur, so that sensor noise does not win."""
    g = cv2.GaussianBlur(to_gray_f32(img), (5, 5), 0)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def pick_reference(frames: list[np.ndarray], mode: str) -> int:
    if mode == "sharpest":
        return int(np.argmax([sharpness(f) for f in frames]))
    return len(frames) // 2


def aligned_median(frames: list[np.ndarray], ref_idx: int, motion: int, levels: int,
                   iters: int, eps: float, min_cc: float) -> tuple[np.ndarray, list[float | None]]:
    ref = frames[ref_idx]
    levels = pyramid_levels(ref.shape, levels)
    h, w = ref.shape[:2]
    c = 1 if ref.ndim == 2 else ref.shape[2]

    layers = [ref.reshape(h, w, c).astype(np.float32)]
    ccs: list[float | None] = [None] * len(frames)
    ccs[ref_idx] = 1.0

    for i, mov in enumerate(frames):
        if i == ref_idx:
            continue
        aligned, valid, cc = align_to_ref(ref, mov, motion, levels, iters, eps, min_cc)
        ccs[i] = cc
        if aligned is None:
            continue
        layer = aligned.reshape(h, w, c).astype(np.float32)
        layer[~valid] = np.nan
        layers.append(layer)

    stack = np.stack(layers, axis=0)
    # Plain median where every frame is valid (fast). NaN-aware median only on
    # the border pixels that some warps left empty. The reference is always
    # valid, so no pixel is all-NaN.
    with np.errstate(invalid="ignore"):
        fused = np.median(stack, axis=0)
    holes = np.isnan(fused)
    if holes.any():
        fused[holes] = np.nanmedian(stack[:, holes], axis=0)

    out = np.clip(np.rint(fused), 0, 255).astype(np.uint8)
    return (out[..., 0] if c == 1 else out), ccs


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_roi(s: str | None) -> tuple[int, int, int, int] | None:
    if not s:
        return None
    parts = [int(v) for v in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--roi must be x,y,w,h")
    return tuple(parts)  # type: ignore[return-value]


def main() -> None:
    ap = argparse.ArgumentParser(description="Temporal fusion with an aligned median.")
    ap.add_argument("src", type=Path, help="directory of images or a video file")
    ap.add_argument("out", type=Path, help="output directory")
    ap.add_argument("--window", type=int, default=9, help="frames per fused output (default 9)")
    ap.add_argument("--stride", type=int, default=None, help="frames between window starts (default = window)")
    ap.add_argument("--roi", type=str, default=None,
                    help="crop x,y,w,h before alignment. Give a margin around the clock face.")
    ap.add_argument("--motion", choices=list(MOTIONS), default="affine",
                    help="alignment model (default affine). Use homography across PTZ preset revisits.")
    ap.add_argument("--ref", choices=["middle", "sharpest"], default="middle",
                    help="reference frame in each window (default middle)")
    ap.add_argument("--min-cc", type=float, default=0.7,
                    help="drop frames whose ECC correlation is below this (default 0.7)")
    ap.add_argument("--levels", type=int, default=3, help="ECC pyramid levels (default 3)")
    ap.add_argument("--iters", type=int, default=100, help="ECC iterations per level (default 100)")
    ap.add_argument("--eps", type=float, default=1e-5, help="ECC stop threshold (default 1e-5)")
    ap.add_argument("--gray", action="store_true", help="fuse in grayscale (smaller, and OCR-ready)")
    args = ap.parse_args()

    if args.window < 2:
        raise SystemExit("--window must be 2 or more")
    stride = args.stride or args.window
    motion = MOTIONS[args.motion]
    roi = parse_roi(args.roi)
    args.out.mkdir(parents=True, exist_ok=True)

    frames_iter = iter_frames(args.src, roi)
    if args.gray:
        frames_iter = ((n, cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for n, f in frames_iter)

    count = 0
    for window in iter_windows(frames_iter, args.window, stride):
        names = [n for n, _ in window]
        frames = [f for _, f in window]
        ref_idx = pick_reference(frames, args.ref)
        fused, ccs = aligned_median(frames, ref_idx, motion, args.levels, args.iters, args.eps, args.min_cc)

        out_path = args.out / f"median_{names[0]}__{names[-1]}.png"
        cv2.imwrite(str(out_path), fused)
        kept = sum(1 for c in ccs if c is not None and c >= args.min_cc)
        cc_txt = " ".join("ref" if i == ref_idx else f"{c:.3f}" for i, c in enumerate(ccs))
        print(f"{out_path.name}: kept {kept}/{len(frames)}  ref={names[ref_idx]}  cc=[{cc_txt}]")
        count += 1

    if count == 0:
        raise SystemExit("no windows produced: check the input path and --window")


if __name__ == "__main__":
    main()