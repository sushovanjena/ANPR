"""Draw the cascade's output on video, for demos and visual checks.

The renderer runs the cascade frame by frame (not paced to real time), draws the
tracked vehicles, their plates and the running consensus text, and writes an
annotated video that a notebook can play inline.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from anpr.pipeline.cascade import Cascade, FrameResult
from anpr.pipeline.consensus import decide
from anpr.pipeline.vehicle import COCO_VEHICLES

_PALETTE = [
    (66, 133, 244),
    (52, 168, 83),
    (251, 188, 5),
    (234, 67, 53),
    (171, 71, 188),
    (0, 172, 193),
]


def track_color(track_id: int) -> tuple[int, int, int]:
    """Return a stable BGR colour for a track id."""
    r, g, b = _PALETTE[track_id % len(_PALETTE)]
    return (b, g, r)


def _label(frame: np.ndarray, text: str, x: int, y: int, color, scale: float) -> None:
    """Draw ``text`` with a filled background whose bottom-left is ``(x, y)``."""
    font, thick = cv2.FONT_HERSHEY_SIMPLEX, max(1, int(round(scale * 2)))
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    y = max(y, th + base)
    cv2.rectangle(frame, (x, y - th - base), (x + tw + 4, y + base // 2), color, -1)
    cv2.putText(frame, text, (x + 2, y - 2), font, scale, (255, 255, 255), thick)


def draw_frame(
    frame: np.ndarray,
    results: list[FrameResult],
    plate_text: dict[int, str] | None = None,
) -> np.ndarray:
    """Return a copy of ``frame`` annotated with the cascade's results.

    Args:
        frame: BGR image.
        results: Per-vehicle results from ``Cascade.process`` for this frame.
        plate_text: Running consensus plate per track id. A track without an
            entry shows only its id and class.

    Returns:
        The annotated copy; the input frame is not modified.
    """
    out = frame.copy()
    scale = max(0.5, out.shape[1] / 1600)
    thick = max(2, int(round(scale * 3)))
    plate_text = plate_text or {}
    for r in results:
        color = track_color(r.track_id)
        x1, y1, x2, y2 = (int(round(v)) for v in r.vehicle_box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        name = COCO_VEHICLES.get(r.vehicle_cls, "vehicle")
        _label(out, f"#{r.track_id} {name}", x1, y1, color, scale)
        if r.plate_box is not None:
            px1, py1, px2, py2 = (int(round(v)) for v in r.plate_box)
            cv2.rectangle(out, (px1, py1), (px2, py2), (0, 255, 255), thick)
        text = plate_text.get(r.track_id)
        if text:
            _label(out, text, x1, y2 + int(28 * scale), (0, 0, 0), scale * 1.1)
    return out


def running_consensus(cascade: Cascade) -> dict[int, str]:
    """Return the current consensus plate text for every track with reads."""
    out = {}
    for tid, reads in cascade.reads.items():
        decision = decide(reads)
        if decision and decision.text:
            out[tid] = decision.text
    return out


def to_h264(path: str | Path) -> Path:
    """Re-encode a video as H.264 so browsers can play it; return the new path.

    Args:
        path: Video written by OpenCV (``mp4v``), which most browsers cannot play.

    Raises:
        RuntimeError: If ``ffmpeg`` fails.
    """
    src = Path(path)
    dst = src.with_name(src.stem + "_h264.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src)]
    cmd += ["-vcodec", "libx264", "-preset", "veryfast", "-crf", "26"]
    cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(done.stderr[-500:])
    return dst


def render_video(
    source: str | Path,
    out_path: str | Path,
    cascade: Cascade,
    max_seconds: float | None = 15.0,
    out_width: int = 960,
) -> dict:
    """Run ``cascade`` over a video file and write an annotated copy.

    Args:
        source: Input video file.
        out_path: Output file (OpenCV ``mp4v``; pass it to ``to_h264`` to play
            it in a browser).
        cascade: Cascade to run; its tracker and reads are updated in place.
        max_seconds: Stop after this much video, or None for the whole file.
        out_width: Width of the written video; height keeps the aspect ratio.

    Returns:
        Counts: ``frames``, ``tracks`` (distinct vehicle tracks drawn) and
        ``plates_read`` (tracks with a consensus plate).

    Raises:
        FileNotFoundError: If the video cannot be opened.
    """
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    limit = int(max_seconds * fps) if max_seconds else None
    writer, size, frames, seen = None, None, 0, set()
    try:
        while limit is None or frames < limit:
            ok, frame = cap.read()
            if not ok:
                break
            results = cascade.process(frame, frames, frames / fps)
            seen.update(r.track_id for r in results)
            shown = draw_frame(frame, results, running_consensus(cascade))
            if writer is None:
                h, w = shown.shape[:2]
                size = (out_width, int(round(h * out_width / w / 2)) * 2)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, size)
            writer.write(cv2.resize(shown, size))
            frames += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    return {
        "frames": frames,
        "tracks": len(seen),
        "plates_read": len(running_consensus(cascade)),
    }


class LiveViewer:
    """Per-frame callback for ``StreamRunner``: show frames live and record them.

    The recording is kept at the source's real time: frames the stream dropped
    are filled by repeating the last drawn frame, so the video lasts as long as
    the stream did.

    Args:
        show: Called with ``(jpeg_bytes, status_text)`` at most every
            ``min_interval`` seconds, for example to update a notebook output.
        out_path: Optional output video (OpenCV ``mp4v``; see ``to_h264``).
        fps: Source frame rate, used for the recording.
        out_width: Width of the shown and recorded frames.
        min_interval: Minimum seconds between ``show`` calls.
    """

    def __init__(
        self,
        show,
        out_path: str | Path | None = None,
        fps: float = 30.0,
        out_width: int = 960,
        min_interval: float = 0.1,
    ) -> None:
        self.show = show
        self.out_path = out_path
        self.fps = fps
        self.out_width = out_width
        self.min_interval = min_interval
        self.cascade: Cascade | None = None
        self.frames_shown = 0
        self._writer = None
        self._last_index = -1
        self._last_frame: np.ndarray | None = None
        self._last_shown = 0.0

    def __call__(self, image: np.ndarray, results: list[FrameResult], index: int):
        """Draw ``results`` on ``image``, show it and add it to the recording."""
        text = running_consensus(self.cascade) if self.cascade else {}
        h, w = image.shape[:2]
        size = (self.out_width, int(round(h * self.out_width / w / 2)) * 2)
        small = cv2.resize(draw_frame(image, results, text), size)
        if self.out_path is not None:
            if self._writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._writer = cv2.VideoWriter(
                    str(self.out_path), fourcc, self.fps, size
                )
            repeat = max(1, index - self._last_index) if self._last_index >= 0 else 1
            if self._last_frame is not None:
                for _ in range(repeat - 1):
                    self._writer.write(self._last_frame)
            self._writer.write(small)
        self._last_index, self._last_frame = index, small
        now = time.perf_counter()
        if now - self._last_shown >= self.min_interval:
            self._last_shown = now
            ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                status = f"frame {index}  |  {len(results)} vehicles  |  plates: "
                status += ", ".join(sorted(set(text.values()))) or "none yet"
                self.show(jpg.tobytes(), status)
                self.frames_shown += 1

    def close(self) -> None:
        """Finish the recording."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None


class GridViewer:
    """Per-frame callback for ``MultiStreamRunner``: a live tile grid of cameras.

    Each camera keeps its latest annotated frame in a tile. The grid is shown at
    most every ``min_interval`` seconds and recorded in real time (each grid is
    held until the next one), so the video lasts as long as the run.

    Args:
        show: Called with ``(jpeg_bytes, status_text)``.
        cascades: Camera id to its ``Cascade``, for the running plate text.
        cols: Tiles per row.
        tile_width: Width of one tile in pixels.
        out_path: Optional output video (OpenCV ``mp4v``; see ``to_h264``).
        fps: Frame rate of the recording.
        min_interval: Minimum seconds between grids.
    """

    def __init__(
        self,
        show,
        cascades: dict[str, Cascade],
        cols: int = 2,
        tile_width: int = 480,
        out_path: str | Path | None = None,
        fps: float = 10.0,
        min_interval: float = 0.15,
    ) -> None:
        self.show = show
        self.cascades = cascades
        self.cols = cols
        self.tile_width = tile_width
        self.out_path = out_path
        self.fps = fps
        self.min_interval = min_interval
        self.grids_shown = 0
        self._tiles: dict[str, np.ndarray] = {}
        self._writer = None
        self._size: tuple[int, int] | None = None
        self._last_shown = 0.0
        self._last_grid: np.ndarray | None = None
        self._debt = 0.0

    def _tile(self, cam_id, image, results) -> np.ndarray:
        text = running_consensus(self.cascades[cam_id])
        h, w = image.shape[:2]
        size = (self.tile_width, int(round(h * self.tile_width / w / 2)) * 2)
        tile = cv2.resize(draw_frame(image, results, text), size)
        label = f"{cam_id}  {len(results)} veh  {len(set(text.values()))} plates"
        _label(tile, label, 4, 22, (40, 40, 40), 0.6)
        return tile

    def _grid(self) -> np.ndarray:
        ids = sorted(self._tiles)
        h = max(t.shape[0] for t in self._tiles.values())
        w = self.tile_width
        rows = []
        for i in range(0, len(ids), self.cols):
            row = []
            for cid in ids[i : i + self.cols]:
                t = self._tiles[cid]
                pad = np.zeros((h, w, 3), np.uint8)
                pad[: t.shape[0]] = t
                row.append(pad)
            while len(row) < self.cols:
                row.append(np.zeros((h, w, 3), np.uint8))
            rows.append(np.hstack(row))
        return np.vstack(rows)

    def __call__(self, cam_id: str, image: np.ndarray, results, index: int):
        """Update ``cam_id``'s tile and, if due, show and record the grid."""
        self._tiles[cam_id] = self._tile(cam_id, image, results)
        now = time.perf_counter()
        if now - self._last_shown < self.min_interval:
            return
        grid = self._grid()
        if self.out_path is not None:
            if self._writer is None:
                self._size = (grid.shape[1], grid.shape[0])
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._writer = cv2.VideoWriter(
                    str(self.out_path), fourcc, self.fps, self._size
                )
            if self._last_grid is not None and self._last_grid.shape == grid.shape:
                self._debt += (now - self._last_shown) * self.fps
                for _ in range(max(0, int(self._debt) - 1)):
                    self._writer.write(self._last_grid)
                self._debt -= int(self._debt)
            self._writer.write(grid)
        self._last_grid, self._last_shown = grid, now
        ok, jpg = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            plates = sum(
                len(set(running_consensus(c).values())) for c in self.cascades.values()
            )
            self.show(jpg.tobytes(), f"{len(self._tiles)} cameras  |  {plates} plates")
            self.grids_shown += 1

    def close(self) -> None:
        """Finish the recording."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
