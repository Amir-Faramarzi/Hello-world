import cv2
import numpy as np
import sys


def compute_total_redness(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    ret, frame = cap.read()
    if not ret:
        raise IOError("Could not read first frame from video")

    # Let user select ROI on the first frame
    roi = cv2.selectROI("Select ROI and press ENTER", frame, showCrosshair=True)
    x, y, w, h = roi
    cv2.destroyWindow("Select ROI and press ENTER")

    # Reset to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    total_redness = 0.0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        roi_frame = frame[y:y+h, x:x+w]
        # BGR format in OpenCV
        b = roi_frame[:, :, 0].astype(np.int16)
        g = roi_frame[:, :, 1].astype(np.int16)
        r = roi_frame[:, :, 2].astype(np.int16)
        redness = r - np.maximum(g, b)
        redness[redness < 0] = 0
        total_redness += np.sum(redness)
    cap.release()
    return total_redness, frame_count


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python redness_intensity.py <video_path>")
        sys.exit(1)
    video_path = sys.argv[1]
    total_redness, nframes = compute_total_redness(video_path)
    print(f"Processed {nframes} frames")
    print(f"Total redness intensity in ROI: {total_redness}")

