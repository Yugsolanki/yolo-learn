#!/usr/bin/env python3
"""
Frame Capture Tool
==================

Captures frames from a webcam or video file over a 60-second window,
triggered by the spacebar. Supports multiple frame-sampling strategies.

Each spacebar press creates a NEW subfolder inside the base output
directory, named `NNNN_xxxx` where NNNN is a zero-padded sequential
index and xxxx is a random 4-char suffix. This guarantees:
  * uniqueness across runs (random suffix),
  * collision resistance (retry on FileExistsError),
  * natural chronological ordering when listing the base directory.

Usage:
    python frame_capture.py [video_path] [options]

Example:
    Webcam + 15 evenly-distributed frames:
    python frame_capture.py -m evenly_distributed -n 15
    
    Video file + maximum consecutive frames:
    python frame_capture.py /path/to/clip.mp4 -m consecutive -n max -o out_run1
    
    # Webcam, capture every frame for 60s (max + consecutive is the only sensible pairing)
    python frame_capture.py -n max
    
    # 10 evenly distributed frames, custom output folder, 90s window
    python frame_capture.py -n 10 -d 90 -o frames/session_a
    
    # 30 consecutive frames from a video file, with live preview
    python frame_capture.py clip.mp4 -m consecutive -n 30 --preview

See `--help` for full CLI reference.
"""

from __future__ import annotations

import argparse
import secrets
import string
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CAPTURE_DURATION_SEC = 60.0          # Length of the capture window.
DEFAULT_OUTPUT_DIR = "captured_frames"
DEFAULT_JPEG_QUALITY = 95            # 0-100; higher = better quality.
TRIGGER_KEY = ord(" ")               # Spacebar.
QUIT_KEY = ord("q")
COUNTDOWN_UPDATE_INTERVAL = 0.25     # Seconds between console countdown prints.

SESSION_ID_ALPHABET = string.ascii_lowercase + string.digits
SESSION_ID_LENGTH = 4                # e.g. "a7k2"
SESSION_PREFIX_WIDTH = 4             # zero-padded counter, e.g. "0001"


@dataclass
class CaptureConfig:
    """Runtime options controlling how frames are sampled during capture."""

    mode: str                # "evenly_distributed" | "consecutive"
    count: Optional[int]     # Number of frames, or None for "max".
    base_dir: Path           # Base directory containing per-session folders.
    jpeg_quality: int
    duration: float = CAPTURE_DURATION_SEC


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Capture frames from a webcam or video file over 60 seconds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "video",
        nargs="?",
        default=None,
        help="Path to a video file. If omitted, the default webcam (index 0) is used.",
    )
    parser.add_argument(
        "-m", "--mode",
        choices=("evenly_distributed", "consecutive"),
        default="evenly_distributed",
        help="Frame distribution strategy.",
    )
    parser.add_argument(
        "-n", "--count",
        default="max",
        help="Number of frames to capture (integer) or 'max' to capture as "
             "many as the source permits.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Base directory. A new subfolder (NNNN_xxxx) is created per "
             "spacebar press.",
    )
    parser.add_argument(
        "-q", "--quality",
        type=int,
        default=DEFAULT_JPEG_QUALITY,
        help="JPEG quality (0-100).",
    )
    parser.add_argument(
        "-d", "--duration",
        type=float,
        default=CAPTURE_DURATION_SEC,
        help="Capture duration in seconds.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show a live preview window during the capture window.",
    )
    return parser.parse_args(argv)


def resolve_count(raw: str) -> Optional[int]:
    """Return None for 'max', otherwise an int >= 1."""
    if str(raw).lower() == "max":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--count must be 'max' or a positive integer (got {raw!r})"
        ) from exc
    if value < 1:
        raise argparse.ArgumentTypeError("--count must be >= 1")
    return value


# ---------------------------------------------------------------------------
# Session directory management
# ---------------------------------------------------------------------------

def ensure_base_dir(path: Path) -> None:
    """Create the base output directory (and parents), raising on failure."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Could not create base directory {path}: {exc}") from exc


def _next_session_index(base_dir: Path) -> int:
    """
    Return the next available numeric prefix by scanning existing
    subfolders of the form NNNN_xxxx. Starts at 1 if none exist.
    """
    highest = 0
    try:
        for entry in base_dir.iterdir():
            if not entry.is_dir():
                continue
            prefix = entry.name.split("_", 1)[0]
            if prefix.isdigit():
                highest = max(highest, int(prefix))
    except OSError:
        # If we can't read the dir we just start from 1 and let the
        # FileExistsError path below handle any collision.
        pass
    return highest + 1


def _unique_session_dir(base_dir: Path) -> Path:
    """
    Create and return a fresh session directory under `base_dir`.

    Name format:  NNNN_xxxx
        NNNN  = zero-padded sequential index (so folders sort in
                creation order in any alphabetical listing)
        xxxx  = random 4-char [a-z0-9] suffix (uniqueness across runs)

    The retry loop guards against the vanishingly unlikely case where
    both the sequential prefix AND the random suffix collide with an
    existing folder.
    """
    index = _next_session_index(base_dir)
    for _ in range(50):  # 50 attempts is essentially certain to succeed
        suffix = "".join(
            secrets.choice(SESSION_ID_ALPHABET)
            for _ in range(SESSION_ID_LENGTH)
        )
        candidate = base_dir / f"{index:0{SESSION_PREFIX_WIDTH}d}_{suffix}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            index += 1  # bump prefix, keep trying
        except OSError as exc:
            raise RuntimeError(
                f"Could not create session directory under {base_dir}: {exc}"
            ) from exc
    raise RuntimeError(
        f"Failed to create a unique session directory under {base_dir} "
        f"after 50 attempts."
    )


# ---------------------------------------------------------------------------
# Video source helpers
# ---------------------------------------------------------------------------

def open_source(video_path: Optional[str]) -> cv2.VideoCapture:
    """Open the webcam or a video file, raising on failure."""
    if video_path is None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open the default webcam (index 0).")
        print("[INFO] Opened default webcam (index 0).")
    else:
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {path}")
        print(f"[INFO] Opened video file: {path}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


# ---------------------------------------------------------------------------
# Frame capture
# ---------------------------------------------------------------------------

def save_frame(frame, out_dir: Path, index: int, quality: int) -> Path:
    """Write a frame to disk as JPEG and return the file path."""
    filename = out_dir / f"frame_{index:05d}.jpg"
    ok = cv2.imwrite(
        str(filename),
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for {filename}")
    return filename


def capture_max_consecutive(
    cap: cv2.VideoCapture,
    cfg: CaptureConfig,
    session_dir: Path,
    show_preview: bool,
) -> int:
    """Capture every available frame for `cfg.duration` seconds."""
    start = time.monotonic()
    last_status = start
    count = 0
    fps_estimate = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"[INFO] Capturing MAX consecutive frames for {cfg.duration:.0f}s "
          f"(source FPS ~{fps_estimate:.1f}).")

    while True:
        elapsed = time.monotonic() - start
        if elapsed >= cfg.duration:
            break

        ok, frame = cap.read()
        if not ok:
            print("[WARN] Stream ended before capture window completed.")
            break

        count += 1
        save_frame(frame, session_dir, count, cfg.jpeg_quality)

        now = time.monotonic()
        if now - last_status >= COUNTDOWN_UPDATE_INTERVAL:
            remaining = max(0.0, cfg.duration - elapsed)
            print(f"\r[CAPTURE] frames={count:5d}  remaining={remaining:5.1f}s",
                  end="", flush=True)
            last_status = now

        if show_preview:
            _draw_overlay(frame, count, cfg.duration - elapsed)
            cv2.imshow("Capturing (space=trigger, q=quit)", frame)
            if cv2.waitKey(1) & 0xFF == QUIT_KEY:
                print("\n[INFO] User quit during capture.")
                break

    print()
    return count


def capture_evenly_distributed(
    cap: cv2.VideoCapture,
    cfg: CaptureConfig,
    session_dir: Path,
    show_preview: bool,
) -> int:
    """
    Sample `cfg.count` frames at equal intervals across `cfg.duration`.

    Timing logic
    ------------
    If N frames are requested over duration D seconds, the sampling interval
    is I = D / N seconds. Targets are:

        t_k = k * I          for k = 0, 1, ..., N-1

    For a file source we sleep until t_k before reading. For a webcam we
    read frames continuously and save the first frame at or after t_k.
    """
    assert cfg.count is not None
    n = cfg.count
    duration = cfg.duration
    interval = duration / n
    targets = [k * interval for k in range(n)]

    print(f"[INFO] Capturing {n} frames evenly distributed across "
          f"{duration:.0f}s (interval = {interval:.2f}s).")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    is_file = src_fps and src_fps > 1.0

    start = time.monotonic()
    saved = 0
    last_status = start

    for target in targets:
        while True:
            if is_file:
                wait = (start + target) - time.monotonic()
                if wait > 0:
                    time.sleep(wait)

            ok, frame = cap.read()
            if not ok:
                print("\n[WARN] Stream ended during capture.")
                return saved

            elapsed = time.monotonic() - start
            if elapsed >= target or not is_file:
                break

        saved += 1
        save_frame(frame, session_dir, saved, cfg.jpeg_quality)

        now = time.monotonic()
        if now - last_status >= COUNTDOWN_UPDATE_INTERVAL or saved == n:
            remaining = max(0.0, duration - (now - start))
            print(f"\r[CAPTURE] frames={saved:5d}/{n}  remaining={remaining:5.1f}s",
                  end="", flush=True)
            last_status = now

        if show_preview:
            _draw_overlay(frame, saved, duration - (time.monotonic() - start))
            cv2.imshow("Capturing (space=trigger, q=quit)", frame)
            if cv2.waitKey(1) & 0xFF == QUIT_KEY:
                print("\n[INFO] User quit during capture.")
                return saved

    print()
    return saved


def capture_consecutive(
    cap: cv2.VideoCapture,
    cfg: CaptureConfig,
    session_dir: Path,
    show_preview: bool,
) -> int:
    """Capture N consecutive frames back-to-back starting from the trigger."""
    assert cfg.count is not None
    n = cfg.count
    print(f"[INFO] Capturing {n} consecutive frames back-to-back.")
    saved = 0
    start = time.monotonic()

    while saved < n:
        ok, frame = cap.read()
        if not ok:
            print("\n[WARN] Stream ended before target count reached.")
            break

        saved += 1
        save_frame(frame, session_dir, saved, cfg.jpeg_quality)

        remaining = max(0.0, cfg.duration - (time.monotonic() - start))
        print(f"\r[CAPTURE] frames={saved:5d}/{n}  remaining={remaining:5.1f}s",
              end="", flush=True)

        if show_preview:
            _draw_overlay(frame, saved, remaining)
            cv2.imshow("Capturing (space=trigger, q=quit)", frame)
            if cv2.waitKey(1) & 0xFF == QUIT_KEY:
                print("\n[INFO] User quit during capture.")
                return saved

    print()
    return saved


def _draw_overlay(frame, saved: int, remaining: float) -> None:
    """Draw a small countdown overlay on the frame (in-place)."""
    text = f"saved={saved}  remaining={remaining:4.1f}s"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2, cv2.LINE_AA)


def run_capture(
    cap: cv2.VideoCapture,
    cfg: CaptureConfig,
    session_dir: Path,
    show_preview: bool,
) -> int:
    """Dispatch to the appropriate capture strategy for this session."""
    if cfg.count is None:
        return capture_max_consecutive(cap, cfg, session_dir, show_preview)
    if cfg.mode == "evenly_distributed":
        return capture_evenly_distributed(cap, cfg, session_dir, show_preview)
    return capture_consecutive(cap, cfg, session_dir, show_preview)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        count = resolve_count(args.count)
    except argparse.ArgumentTypeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    base_dir = Path(args.output_dir).expanduser().resolve()
    cfg = CaptureConfig(
        mode=args.mode,
        count=count,
        base_dir=base_dir,
        jpeg_quality=max(0, min(100, args.quality)),
        duration=args.duration,
    )

    # Set up the base directory once; per-session folders are created
    # fresh on every trigger.
    try:
        ensure_base_dir(cfg.base_dir)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    try:
        cap = open_source(args.video)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    window_name = "Frame Capture — press SPACE to trigger, q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print(f"[INFO] Base output directory: {cfg.base_dir}")
    print(f"[INFO] Mode: {cfg.mode} | Count: {cfg.count or 'max'} "
          f"| Duration: {cfg.duration:.0f}s")
    print("[INFO] Press SPACE to begin capture. Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[INFO] End of stream.")
                break

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == QUIT_KEY:
                print("[INFO] Quit requested.")
                break

            if key == TRIGGER_KEY:
                print(f"\n[INFO] Capture triggered at "
                      f"{time.strftime('%Y-%m-%d %H:%M:%S')}")

                # Create a fresh session subfolder for THIS trigger.
                try:
                    session_dir = _unique_session_dir(cfg.base_dir)
                except RuntimeError as exc:
                    print(f"[ERROR] {exc}", file=sys.stderr)
                    continue  # don't kill the loop; user can retry

                print(f"[INFO] Session folder: {session_dir}")

                saved = run_capture(cap, cfg, session_dir,
                                    show_preview=args.preview)
                print(f"[INFO] Capture complete: {saved} frame(s) saved to "
                      f"{session_dir}")
                print("[INFO] Press SPACE to capture again, q to quit.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())