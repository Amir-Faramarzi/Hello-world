import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


def compute_total_redness(video_path, roi):
    """Compute summed redness intensity within ROI across all frames."""
    x, y, w, h = roi
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    total_redness = 0.0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        roi_frame = frame[y:y + h, x:x + w]
        b = roi_frame[:, :, 0].astype(np.int16)
        g = roi_frame[:, :, 1].astype(np.int16)
        r = roi_frame[:, :, 2].astype(np.int16)
        redness = r - np.maximum(g, b)
        redness[redness < 0] = 0
        total_redness += np.sum(redness)

    cap.release()
    return total_redness, frame_count


class RednessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Redness Intensity Calculator")
        self.video_path = None
        self.frame = None
        self.roi = None
        self.tkimg = None
        self.start_x = self.start_y = 0
        self.rect = None

        self.canvas = tk.Canvas(root)
        self.canvas.pack()

        control_frame = tk.Frame(root)
        control_frame.pack(pady=5)
        tk.Button(control_frame, text="Open Video", command=self.open_video).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Compute Redness", command=self.compute_redness).pack(side=tk.LEFT, padx=5)
        self.result_var = tk.StringVar()
        tk.Label(control_frame, textvariable=self.result_var).pack(side=tk.LEFT, padx=5)

    def open_video(self):
        path = filedialog.askopenfilename(title="Select video file",
                                          filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"),
                                                     ("All files", "*.*")])
        if not path:
            return
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            messagebox.showerror("Error", "Failed to open video")
            return
        self.video_path = path
        self.frame = frame
        self.display_frame()

    def display_frame(self):
        frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        self.tkimg = ImageTk.PhotoImage(img)
        self.canvas.config(width=img.width, height=img.height)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tkimg)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def on_press(self, event):
        self.start_x, self.start_y = event.x, event.y
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red")

    def on_drag(self, event):
        if not self.rect:
            return
        self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_release(self, event):
        if not self.rect:
            return
        x0, y0, x1, y1 = self.canvas.coords(self.rect)
        x = int(min(x0, x1))
        y = int(min(y0, y1))
        w = int(abs(x1 - x0))
        h = int(abs(y1 - y0))
        self.roi = (x, y, w, h)

    def compute_redness(self):
        if not self.video_path or not self.roi:
            messagebox.showwarning("Warning", "Open a video and select a region first")
            return
        try:
            total, frames = compute_total_redness(self.video_path, self.roi)
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return
        self.result_var.set(f"Frames: {frames}  Redness: {total:.2f}")


def main():
    root = tk.Tk()
    app = RednessApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
