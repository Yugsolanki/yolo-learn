"""
temporal_fusion_median.py
Align N frames of a static clock face to a reference, then fuse via per-pixel median.

Usage:
    python temporal_fusion_median.py --input ./frames --output fused_median.png --max-frames 25
"""

import argparse
import glob
import os

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def align_ecc(src_gray, ref_gray, motion="affine", max_iter=100, eps=1e-5):
    """Align src to ref using ECC image alignment. Returns warp matrix or None."""
    motion_map = {
        "translation": cv2.MOTION_TRANSLATION,
        "euclidean":   cv2.MOTION_EUCLIDEAN,
        "affine":      cv2.MOTION_AFFINE,
        "homography":  cv2.MOTION_HOMOGRAPHY,
    }
    mode = motion_map[motion]
    warp = np.eye(2, 3, dtype=np.float32) if mode != cv2.MOTION_HOMOGRAPHY \
           else np.eye(3, 3, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)
    try:
        _, warp = cv2.findTransformECC(ref_gray, src_gray, warp, mode, criteria,
                                       None, 5)  # gaussFiltSize=5 helps noisy CCTV
        return warp
    except cv2.error:
        return None


def align_orb_fallback(src_gray, ref_gray):
    """Feature-based fallback when ECC fails (large motion / very low texture)."""
    orb = cv2.ORB_create(nfeatures=2000)
    k1, d1 = orb.detectAndCompute(src_gray, None)
    k2, d2 = orb.detectAndCompute(ref_gray, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    matches = sorted(matches, key=lambda m: m.distance)[:100]
    if len(matches) < 8:
        return None
    pts1 = np.float32([k1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([k2[m.trainIdx].pt for m in matches])
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    if H is None or mask is None or mask.sum() < 6:
        return None
    return H  # 3x3 homography


def warp_frame(img, warp, shape):
    h, w = shape
    if warp is None:
        return img
    if warp.shape == (3, 3):
        return cv2.warpPerspective(img, warp, (w, h),
                                   flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                   borderMode=cv2.BORDER_CONSTANT)
    return cv2.warpAffine(img, warp, (w, h),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def load_frames(input_path, max_frames=None):
    if os.path.isdir(input_path):
        paths = sorted(
            p for p in glob.glob(os.path.join(input_path, "*"))
            if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        )
    else:
        paths = [input_path]
    if max_frames:
        paths = paths[:max_frames]
    frames = [cv2.imread(p, cv2.IMREAD_COLOR) for p in paths]
    frames = [f for f in frames if f is not None]
    if len(frames) < 2:
        raise RuntimeError("Need at least 2 readable frames.")
    return frames, paths


def valid_overlap_mask(ref_shape, warps):
    """Build a mask of pixels valid in ALL frames (excludes black warp borders)."""
    h, w = ref_shape
    mask = np.full((h, w), 255, dtype=np.uint8)
    for warp in warps:
        if warp is None:
            continue
        frame_mask = np.full((h, w), 255, dtype=np.uint8)
        if warp.shape == (3, 3):
            wm = cv2.warpPerspective(frame_mask, warp, (w, h),
                                     flags=cv2.WARP_INVERSE_MAP,
                                     borderMode=cv2.BORDER_CONSTANT)
        else:
            wm = cv2.warpAffine(frame_mask, warp, (w, h),
                                flags=cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_CONSTANT)
        mask = cv2.bitwise_and(mask, wm)
    # Erode to kill 1-2 px interpolation halos at edges
    return cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Folder of frames or single image")
    ap.add_argument("--output", default="fused_median.png")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--motion", default="affine",
                    choices=["translation", "euclidean", "affine", "homography"],
                    help="PTZ drift is usually small: euclidean/affine is enough")
    ap.add_argument("--crop-to-overlap", action="store_true",
                    help="Crop output to the region valid in every frame")
    ap.add_argument("--downscale", type=float, default=1.0,
                    help="Alignment-only downscale factor (e.g. 0.5 for speed)")
    args = ap.parse_args()

    frames, paths = load_frames(args.input, args.max_frames)
    print(f"[info] Loaded {len(frames)} frames")

    ref = frames[len(frames) // 2]           # middle frame as reference
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    if args.downscale != 1.0:
        small = cv2.resize(ref_gray, None, fx=args.downscale, fy=args.downscale)
        ref_align = small
    else:
        ref_align = ref_gray

    aligned, warps = [], []
    for i, f in enumerate(frames):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        g_align = (cv2.resize(g, None, fx=args.downscale, fy=args.downscale)
                   if args.downscale != 1.0 else g)

        warp = align_ecc(g_align, ref_align, motion=args.motion)
        if warp is None:
            warp = align_orb_fallback(g_align, ref_align)
        if warp is None:
            print(f"[warn] Frame {i} ({paths[i]}) failed to align — skipped")
            continue

        # Rescale warp back to full resolution if we aligned on downscaled images
        if args.downscale != 1.0:
            s = 1.0 / args.downscale
            if warp.shape == (3, 3):
                S = np.diag([s, s, 1.0]); warp = S @ warp @ np.linalg.inv(S)
            else:
                warp_full = np.eye(3, 3); warp_full[:2] = warp
                S = np.diag([s, s, 1.0]); warp_full = S @ warp_full @ np.linalg.inv(S)
                warp = warp_full[:2]

        aligned.append(warp_frame(f, warp, ref_gray.shape))
        warps.append(warp)

    if len(aligned) < 2:
        raise RuntimeError("Too few frames aligned successfully.")

    stack = np.stack(aligned, axis=0)                     # (N, H, W, 3)
    fused = np.median(stack, axis=0).astype(np.uint8)     # per-pixel, per-channel median

    if args.crop_to_overlap:
        mask = valid_overlap_mask(ref_gray.shape, warps)
        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            fused = fused[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    cv2.imwrite(args.output, fused)
    print(f"[done] Fused {len(aligned)} frames -> {args.output}")


if __name__ == "__main__":
    main()