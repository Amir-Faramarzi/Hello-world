"""Tkinter views for redness tracker."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable


class MainView(ttk.Frame):
    """Main application window composed of canvas, side panel and timeline."""

    def __init__(self, master: tk.Misc, controller: "Callable[[str, dict[str, object]], None]") -> None:
        super().__init__(master)
        self.controller = controller
        self.pack(fill=tk.BOTH, expand=True)
        self._build_widgets()

    def _build_widgets(self) -> None:
        # Left canvas displaying video
        self.canvas = tk.Canvas(self, bg="black")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right side notebook
        side = ttk.Notebook(self)
        side.pack(side=tk.RIGHT, fill=tk.Y)

        # ROI selection tab
        roi_tab = ttk.Frame(side)
        side.add(roi_tab, text="ROI")
        self.roi_vars = [tk.IntVar(value=0) for _ in range(4)]
        labels = ["x", "y", "w", "h"]
        for i, (var, label) in enumerate(zip(self.roi_vars, labels)):
            ttk.Label(roi_tab, text=label).grid(row=i, column=0, sticky=tk.W)
            ttk.Spinbox(roi_tab, from_=0, to=10000, textvariable=var).grid(row=i, column=1)

        # Noise reduction tab (simplified)
        noise_tab = ttk.Frame(side)
        side.add(noise_tab, text="Noise")
        self.s_min = tk.DoubleVar(value=0.5)
        ttk.Scale(noise_tab, from_=0, to=1, variable=self.s_min).pack(fill=tk.X)
        self.use_max_red = tk.BooleanVar(value=False)
        ttk.Checkbutton(noise_tab, text="Use Max Red Frame", variable=self.use_max_red, command=self._toggle_max_red).pack(anchor=tk.W)

        # Bottom controls
        bottom = ttk.Frame(self)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        self.timeline = ttk.Scale(bottom, from_=0, to=100, orient=tk.HORIZONTAL)
        self.timeline.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.time_label = ttk.Label(bottom, text="00:00 / 00:00")
        self.time_label.pack(side=tk.LEFT)
        self.jump_var = tk.IntVar(value=0)
        ttk.Entry(bottom, textvariable=self.jump_var, width=5).pack(side=tk.LEFT)

    def _toggle_max_red(self) -> None:
        self.controller("toggle_max_red", {"state": self.use_max_red.get()})
