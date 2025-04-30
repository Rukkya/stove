import os
import time
from typing import Any, List, Tuple, Dict
import cv2
import numpy as np
import torch
import asyncio
import streamlit as st
from tempfile import NamedTemporaryFile

from cv2 import Mat
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Paths
weights_path = os.path.join(os.path.dirname(__file__), "yolo-weights.pt")
if not os.path.exists(weights_path):
    raise FileNotFoundError("YOLO weights file not found. Please ensure the weights file is available.")

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
person_detector = YOLO(weights_path).to(device).half()
tracker = DeepSort(max_age=5)

async def process_frame(
    frame: Mat | np.ndarray[Any, np.dtype],
    track_person_map: Dict[int, int],
    previous_positions: Dict[int, Tuple[int, int, int, int]],
) -> Tuple[List[Dict[str, Any]], Dict[int, int], Dict[int, Tuple[int, int, int, int]], int]:
    detections = []
    # Step 1: Detect all persons using YOLO
    results = person_detector(frame)[0]  # Single frame
    for box in results.boxes:
        cls = int(box.cls[0].item())
        if cls != 0:  # Person class only
            continue
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

    # Step 2: Track detected persons using DeepSORT
    tracks = tracker.update_tracks(detections, frame=frame)

    results_out = []
    for tr in tracks:
        if not tr.is_confirmed():
            continue

        track_id = tr.track_id
        l, t, w, h = tr.to_ltwh()
        x1, y1, x2, y2 = int(l), int(t), int(l + w), int(t + h)

        # Calculate movement
        previous_position = previous_positions.get(track_id)
        if previous_position:
            px1, py1, px2, py2 = previous_position
            dx = abs(x1 - px1) + abs(x2 - px2)
            dy = abs(y1 - py1) + abs(y2 - py2)
            movement_threshold = 10  # Adjust as needed
            movement_label = "Moving" if dx + dy > movement_threshold else "Not Moving"
        else:
            movement_label = "Not Moving"

        previous_positions[track_id] = (x1, y1, x2, y2)

        results_out.append({
            "track_id": track_id,
            "bbox": (x1, y1, x2, y2),
            "movement_label": movement_label
        })

    return results_out, track_person_map, previous_positions, len(results)

async def track(video_path: str) -> None:
    track_person_map: Dict[int, int] = {}
    previous_positions: Dict[int, Tuple[int, int, int, int]] = {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"[ERROR] Cannot open video: {video_path}")
        return

    FRAME_SKIP = 5
    frame_count = 0
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    effective_fps = video_fps / FRAME_SKIP if FRAME_SKIP > 0 else video_fps
    delay_per_frame = 1.0 / effective_fps

    st_frame = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        frame = cv2.resize(frame, (640, 640))
        results_out, track_person_map, previous_positions, count = await process_frame(frame, track_person_map, previous_positions)

        for result in results_out:
            x1, y1, x2, y2 = result['bbox']
            movement_label = result['movement_label']
            color = (0, 255, 0) if movement_label == "Moving" else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, movement_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, f"Person Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Convert frame to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(frame_rgb, channels="RGB")
        
        time.sleep(delay_per_frame)

    cap.release()

# === Streamlit UI ===
st.title("Person Tracking and Movement Detection")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    with NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_video_path = tmp_file.name

    if st.button("Start Tracking"):
        asyncio.run(track(tmp_video_path))
