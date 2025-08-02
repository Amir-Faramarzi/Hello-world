"""Application controller tying together model and view."""
from __future__ import annotations

import threading
from pathlib import Path
from queue import Queue

import tkinter as tk
import numpy as np
from numpy.typing import NDArray
from typing import Callable, cast

ArrayU8 = NDArray[np.uint8]

from ..model.algorithms import RedMask, RunningMedian
from ..model.config import Settings, configure_logging
from ..model.video_processing import VideoProcessor
from ..view.main_view import MainView


class MainController:
    """Main controller responsible for coordinating the application."""

    def __init__(self, root: tk.Tk) -> None:
        configure_logging(Path("logs"))
        self.root = root
        self.settings = Settings.load(Path("config/settings.yaml"))
        self.view = MainView(root, self.handle_event)
        self.frame_queue: Queue[tuple[int, ArrayU8]] = Queue()
        self.result_queue: Queue[tuple[int, ArrayU8]] = Queue()
        self.stop_event = threading.Event()
        self.processor = VideoProcessor(
            self.frame_queue,
            self.result_queue,
            [cast(Callable[[ArrayU8], ArrayU8], RedMask()), RunningMedian().update],
            self.stop_event,
        )
        self.processor.start()
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    def handle_event(self, name: str, data: dict[str, object]) -> None:
        if name == "toggle_max_red":
            self.settings.use_max_red = bool(data.get("state"))

    def on_close(self) -> None:
        self.stop_event.set()
        self.processor.join(timeout=1)
        self.settings.save(Path("config/settings.yaml"))
        self.root.destroy()
