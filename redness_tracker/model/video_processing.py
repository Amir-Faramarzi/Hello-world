"""Video processing utilities and trackers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple
from queue import Queue, Empty
import threading

import cv2
import numpy as np
from numpy.typing import NDArray

ArrayU8 = NDArray[np.uint8]

from .algorithms import RedMask


class VideoProcessor(threading.Thread):
    """Background thread applying a sequence of processors to frames.

    The thread reads frames from ``frame_queue`` and writes processed frames to
    ``result_queue``.  Frames are expected to be tuples ``(index, frame)`` where
    ``index`` is the frame number to keep ordering information.
    """

    def __init__(
        self,
        frame_queue: Queue[tuple[int, ArrayU8]],
        result_queue: Queue[tuple[int, ArrayU8]],
        processors: list[Callable[[ArrayU8], ArrayU8]],
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.processors = processors
        self.stop_event = stop_event

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                idx, frame = self.frame_queue.get(timeout=0.1)
            except Empty:
                continue
            for proc in self.processors:
                frame = proc(frame)
            self.result_queue.put((idx, frame))
            self.frame_queue.task_done()


@dataclass
class MaxRedFrameFinder:
    """Find the frame with the maximum red dominance within a region of interest."""

    roi: Tuple[int, int, int, int]
    redmask: RedMask

    def __post_init__(self) -> None:
        self.max_value = -np.inf
        self.index = -1
        self.current = 0

    def update(self, frame: ArrayU8) -> None:
        """Update the statistics with a new frame."""
        x, y, w, h = self.roi
        roi_frame = frame[y : y + h, x : x + w]
        r = roi_frame[:, :, 2].astype(np.int32)
        g = roi_frame[:, :, 1].astype(np.int32)
        b = roi_frame[:, :, 0].astype(np.int32)
        value = np.sum(r - np.maximum(g, b))
        if value > self.max_value:
            self.max_value = value
            self.index = self.current
        self.current += 1

    def max_index(self) -> int:
        """Return the index of the frame with maximal redness."""
        return self.index


@dataclass
class WaveEvent:
    """Event emitted by :class:`WaveTracker`."""

    t_start: int
    t_end: int
    red_intensity: float


class WaveTracker:
    """Track wave like red patterns across the frame.

    The tracker divides the region of interest into sections and sub-bands.  Each
    sub-band maintains an internal state machine (IDLE → TRACKING).  When the top
    row of a sub-band exhibits red coverage of at least 95 %, tracking starts and
    the bottom row pattern is memorised.  Tracking ends when the correlation or
    cosine similarity between the memorised pattern and the current bottom row is
    at least 0.9, upon which an event is emitted.
    """

    def __init__(self, roi: Tuple[int, int, int, int], redmask: RedMask, sections: int = 4, subbands: int = 3) -> None:
        self.roi = roi
        self.redmask = redmask
        self.sections = sections
        self.subbands = subbands
        self.state = [["IDLE" for _ in range(subbands)] for _ in range(sections)]
        self.pattern = [[None for _ in range(subbands)] for _ in range(sections)]
        self.t_start = [[0 for _ in range(subbands)] for _ in range(sections)]
        self.events:  list [WaveEvent] = []

    def update(self, frame: ArrayU8, index: int) -> None:
        x, y, w, h = self.roi
        roi_frame = frame[y : y + h, x : x + w]
        section_width = w // self.sections
        for i in range(self.sections):
            section = roi_frame[:, i * section_width : (i + 1) * section_width]
            sub_height = section.shape[0] // self.subbands
            for j in range(self.subbands):
                sub = section[j * sub_height : (j + 1) * sub_height]
                mask = self.redmask(sub)
                top = mask[0]
                bottom = mask[-1].astype(np.float32)
                coverage = float(np.count_nonzero(top)) / float(top.size)
                if self.state[i][j] == "IDLE" and coverage >= 0.95:
                    self.state[i][j] = "TRACKING"
                    self.pattern[i][j] = bottom
                    self.t_start[i][j] = index
                elif self.state[i][j] == "TRACKING":
                    ref = self.pattern[i][j]
                    if ref is None or bottom.size == 0:
                        continue
                    r = np.corrcoef(ref, bottom)[0, 1]
                    cos = float(np.dot(ref, bottom) / (np.linalg.norm(ref) * np.linalg.norm(bottom)))
                    if r >= 0.9 or cos >= 0.9:
                        intensity = float(np.sum(sub[:, :, 2] - np.maximum(sub[:, :, 0], sub[:, :, 1])))
                        self.events.append(WaveEvent(self.t_start[i][j], index, intensity))
                        self.state[i][j] = "IDLE"
                        self.pattern[i][j] = None

