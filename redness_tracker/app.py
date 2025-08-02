"""Entry point for redness_tracker GUI application."""
from __future__ import annotations

import tkinter as tk

from .controller.main_controller import MainController


def main() -> None:
    root = tk.Tk()
    root.title("Redness Tracker")
    MainController(root)
    root.mainloop()


if __name__ == "__main__":
    main()
