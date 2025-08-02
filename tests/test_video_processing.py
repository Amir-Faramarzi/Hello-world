import numpy as np
import threading
from queue import Queue

from redness_tracker.model.algorithms import RedMask
from redness_tracker.model.video_processing import MaxRedFrameFinder, WaveTracker, VideoProcessor


def test_max_red_frame_finder() -> None:
    frames = [
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.uint8),
    ]
    frames[1][0, 0] = [0, 0, 255]
    finder = MaxRedFrameFinder((0, 0, 2, 2), RedMask())
    for f in frames:
        finder.update(f)
    assert finder.max_index() == 1


def test_wave_tracker_event() -> None:
    tracker = WaveTracker((0, 0, 4, 6), RedMask())
    frame = np.zeros((6, 4, 3), dtype=np.uint8)
    frame[:] = [0, 0, 255]
    tracker.update(frame, 0)
    tracker.update(frame, 1)
    assert tracker.events


def test_video_processor() -> None:
    frame_q: "Queue[tuple[int, np.ndarray]]" = Queue()
    result_q: "Queue[tuple[int, np.ndarray]]" = Queue()
    stop = threading.Event()

    def invert(frame: np.ndarray) -> np.ndarray:
        return 255 - frame

    vp = VideoProcessor(frame_q, result_q, [invert], stop)
    vp.start()
    frame_q.put((0, np.zeros((1, 1, 3), dtype=np.uint8)))
    frame_q.join()
    stop.set()
    vp.join()
    idx, frame = result_q.get_nowait()
    assert idx == 0
    assert frame[0, 0, 0] == 255
