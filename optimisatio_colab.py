import os
import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForImageClassification
import threading
import csv
import time
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
import gc

# Set device for processing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# Suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# ThreadPoolExecutor for async processing
executor = ThreadPoolExecutor(max_workers=5)

# Set paths for models and CSV
current_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(current_dir, "model")
csv_file = os.path.join(current_dir, "person_tracking_data.csv")

# Lock for CSV writes
csv_lock = threading.Lock()

# Initialize YOLO model without half precision to avoid dtype mismatch
yolo_path = os.path.join(folder_path, "yolo11n.pt")
model = YOLO(yolo_path).to(device)

# Initialize gender classification model
gender_processor = AutoImageProcessor.from_pretrained(
    "rizvandwiki/gender-classification"
)
gender_model = AutoModelForImageClassification.from_pretrained(
    "rizvandwiki/gender-classification"
).to(device)

# Initialize DeepSort tracker
tracker = DeepSort(max_age=70, max_cosine_distance=0.3, n_init=3)

# Ensure CSV headers exist
if not os.path.exists(csv_file):
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "serial_number",
                "tracking_id",
                "gender",
                "age",
                "time_spent",
                "start_time",
                "first_appearance_time",
                "last_appearance_time",
            ]
        )

# Track info
global_serial = 1
track_info = {}


# Predict objects using YOLO
def predict_objects(img, conf=0.4):
    results = model.predict(img, conf=conf, verbose=False)
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append([[x1, y1, x2 - x1, y2 - y1], conf, int(box.cls[0])])
    return detections


# Track objects
def track_objects(detections, frame):
    return tracker.update_tracks(detections, frame=frame)


# Predict gender
def predict_gender(crop):
    try:
        img = cv2.resize(crop, (224, 224))
        tensor = gender_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            output = gender_model(**tensor)
        pred_idx = output.logits.argmax(-1).item()
        return gender_model.config.id2label[pred_idx]
    except Exception:
        return "Unknown"


# Update CSV
def update_csv(serial, time_spent):
    with csv_lock:
        rows = []
        with open(csv_file, "r") as file:
            reader = csv.reader(file)
            headers = next(reader)
            for row in reader:
                if int(row[0]) == serial:
                    row[4] = f"{time_spent:.2f}"
                rows.append(row)
        with open(csv_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(rows)


# Process each frame
def process_frame(frame, frame_num, fps):
    global global_serial
    detections = predict_objects(frame)
    tracks = track_objects(detections, frame)

    current_ids = set()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_tlbr())
        crop = frame[y1:y2, x1:x2]
        current_ids.add(track_id)

        if track_id not in track_info:
            gender = predict_gender(crop)
            age = np.random.choice(["25-35", "36-50"])
            track_info[track_id] = {
                "serial_number": global_serial,
                "gender": gender,
                "age": age,
                "start_frame": frame_num,
                "time_spent": 0.0,
                "start_time": now,
                "first_appearance_time": now,
                "last_appearance_time": now,
            }
            global_serial += 1

        elapsed_time = (frame_num - track_info[track_id]["start_frame"]) / fps
        track_info[track_id]["time_spent"] = elapsed_time
        track_info[track_id]["last_appearance_time"] = now

        executor.submit(update_csv, track_info[track_id]["serial_number"], elapsed_time)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID: {track_id}, {track_info[track_id]['gender']}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    # Clean up old tracks
    for tid in list(track_info.keys()):
        if tid not in current_ids:
            del track_info[tid]

    return frame


# Process video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        "output_optimized.mp4", fourcc, fps, (frame_width, frame_height)
    )

    frame_num, start_time = 0, time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame and write output
        result_frame = process_frame(frame, frame_num, fps)
        out.write(result_frame)
        frame_num += 1

        # Display FPS
        if frame_num % 10 == 0:
            elapsed = time.time() - start_time
            logger.info(
                f"Processed {frame_num}/{frame_count} frames, FPS: {frame_num / elapsed:.2f}"
            )

        # Free memory
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    total_time = time.time() - start_time
    logger.info(f"Completed video processing in {total_time:.2f} seconds.")


# Main execution
if __name__ == "__main__":
    video_path = "/content/sample.mp4"
    process_video(video_path)
