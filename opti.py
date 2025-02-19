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
import random
import gc
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# Initialize ThreadPoolExecutor for async processing
executor = ThreadPoolExecutor(max_workers=10)

# Set paths for models and CSV
current_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.getenv("MODEL_PATH", os.path.join(current_dir, "model"))
csv_file = os.path.join(current_dir, "person_tracking_data.csv")
json_output_file = os.path.join(current_dir, "tracking_output.json")

# Initialize CSV lock for concurrent writes
csv_lock = threading.Lock()

# Initialize variables for tracking
serial_number = 1
male_count = 0
female_count = 0
track_info = {}


# Load YOLO model for object detection
def load_yolo_model():
    yolo_path = os.getenv("YOLO_MODEL_PATH", os.path.join(folder_path, "yolo11n.pt"))
    if not os.path.exists(yolo_path):
        raise FileNotFoundError(f"YOLO model not found at {yolo_path}")
    model = YOLO(yolo_path).to(device)
    return model


try:
    model = load_yolo_model()
    class_names = model.names
except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}")
    raise


# Initialize gender classification model
def load_gender_model():
    try:
        processor = AutoImageProcessor.from_pretrained(
            "rizvandwiki/gender-classification"
        )
        model = AutoModelForImageClassification.from_pretrained(
            "rizvandwiki/gender-classification"
        )
        return processor, model
    except Exception as e:
        logger.error(f"Failed to load gender classification model: {e}")
        raise


gender_processor, gender_model = load_gender_model()

# Initialize DeepSort tracker
tracker = DeepSort(max_age=100, max_cosine_distance=0.3, n_init=3, max_iou_distance=0.7)

# Ensure CSV exists with headers
if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
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
        logger.info(f"Created new CSV file: {csv_file}")


# Function to write JSON output
def write_json_output():
    with open(json_output_file, "w") as json_file:
        json.dump(track_info, json_file, indent=4)
    logger.info(f"Exported tracking data to {json_output_file}")


# Function to store tracking data in CSV
def add_new_entry_to_csv(track_id):
    with csv_lock:
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            data = track_info[track_id]
            writer.writerow(
                [
                    data["serial_number"],
                    track_id,
                    data["gender"],
                    data["age"],
                    data["time_spent"],
                    data["start_time"],
                    data["first_appearance_time"],
                    data["last_appearance_time"],
                ]
            )
            logger.info(f"Added new entry for track_id {track_id} to CSV.")


# Update time spent in CSV
def update_time_spent_in_csv(serial_number, new_time_spent):
    rows = []
    with csv_lock:
        if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
            logger.warning("CSV file is empty; cannot update time.")
            return

        with open(csv_file, mode="r", newline="") as file:
            reader = csv.reader(file)
            headers = next(reader)
            for row in reader:
                if int(row[0]) == serial_number:
                    row[4] = f"{new_time_spent:.2f}"
                    row[7] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                rows.append(row)

        with open(csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(rows)
            logger.info(f"Updated time spent for serial_number {serial_number} in CSV.")


# Predict objects using YOLO
def predict_objects(img, model, classes=[0], conf=0.3):
    try:
        results = model.predict(img, classes=classes, conf=conf)
        detections, boxes = [], []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id, confidence = int(box.cls[0]), box.conf[0]
                detections.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])
                boxes.append((x1, y1, x2, y2))
        return img, detections, boxes
    except Exception as e:
        logger.error(f"Error during object prediction: {e}")
        return img, [], []


# Predict gender
def predict_gender(track_roi):
    try:
        track_roi_resized = cv2.resize(track_roi, (224, 224))
        track_roi_tensor = gender_processor(
            images=track_roi_resized, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = gender_model(**track_roi_tensor)
            predicted_class_idx = outputs.logits.argmax(-1).item()
            return gender_model.config.id2label.get(
                predicted_class_idx, "Unknown Gender"
            )
    except Exception as e:
        logger.error(f"Error during gender prediction: {e}")
        return "Unknown"


# Process each frame
def process_frame(img, frame_number, fps):
    global serial_number, male_count, female_count

    result_img, detections, boxes = predict_objects(img, model, classes=[0], conf=0.5)
    tracks = tracker.update_tracks(detections, frame=img)

    current_tracks = set()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_tlbr())
        track_roi = img[y1:y2, x1:x2]

        if track_id not in track_info:
            predicted_gender = predict_gender(track_roi)
            age = random.choice(["25-35", "36-50"])
            track_info[track_id] = {
                "serial_number": serial_number,
                "gender": predicted_gender,
                "age": age,
                "start_frame": frame_number,
                "time_spent": 0.0,
                "start_time": current_time,
                "first_appearance_time": current_time,
                "last_appearance_time": current_time,
            }

            if predicted_gender == "male":
                male_count += 1
            elif predicted_gender == "female":
                female_count += 1

            serial_number += 1
            add_new_entry_to_csv(track_id)

        elapsed_time = (frame_number - track_info[track_id]["start_frame"]) / fps
        track_info[track_id]["time_spent"] = elapsed_time
        track_info[track_id]["last_appearance_time"] = current_time
        executor.submit(
            update_time_spent_in_csv,
            track_info[track_id]["serial_number"],
            elapsed_time,
        )

        # Draw bounding boxes and labels
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text_lines = [
            f"ID: {track_info[track_id]['serial_number']}",
            f"Gender: {track_info[track_id]['gender']}",
            f"Age: {track_info[track_id]['age']}",
            f"Time: {elapsed_time:.2f}s",
        ]
        for i, line in enumerate(text_lines):
            cv2.putText(
                result_img,
                line,
                (x1, y1 - 10 - i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        current_tracks.add(track_id)

    disappeared_tracks = set(track_info.keys()) - current_tracks
    for track_id in disappeared_tracks:
        del track_info[track_id]

    return result_img


# Process video
def process_video(video_path):
    start_processing_time = time.time()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video stream: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        "Processed_Output.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )

    frame_number = 0
    total_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        result_img = process_frame(frame, frame_number, fps)
        out.write(result_img)
        end_time = time.time()

        processing_time = end_time - start_time
        total_time += processing_time
        current_fps = 1 / processing_time if processing_time > 0 else 0
        logger.info(f"Frame {frame_number} processed. FPS: {current_fps:.2f}")

        frame_number += 1
        cv2.imshow("Inference", result_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    write_json_output()

    end_processing_time = time.time()
    logger.info(
        f"Total processing time: {end_processing_time - start_processing_time:.2f} seconds"
    )


# Main execution
if __name__ == "__main__":
    video_path = os.getenv("VIDEO_PATH", "sample.mp4")
    process_video(video_path)
