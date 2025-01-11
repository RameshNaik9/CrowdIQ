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

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# Initialize ThreadPoolExecutor for async processing
executor = ThreadPoolExecutor(max_workers=5)

# Set paths for model and csv
current_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(current_dir, "model")
csv_file = os.path.join(current_dir, "person_tracking_data.csv")

# Initialize CSV lock for concurrent writes
csv_lock = threading.Lock()

# Initialize variables for tracking
serial_number = 1
male_count = 0
female_count = 0
track_info = {}

# Initialize YOLO model for object detection
yolo_path = os.path.join(folder_path, "yolo11n.pt")
model = YOLO(yolo_path)
class_names = model.names

# Initialize gender classification model
gender_processor = AutoImageProcessor.from_pretrained(
    "rizvandwiki/gender-classification"
)
gender_model = AutoModelForImageClassification.from_pretrained(
    "rizvandwiki/gender-classification"
)

# Age detection model setup
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = [
    "(0-2)",
    "(4-6)",
    "(8-12)",
    "(15-20)",
    "(25-32)",
    "(38-43)",
    "(48-53)",
    "(60-100)",
]
ageProto = os.path.join(folder_path, "age_deploy.prototxt")
ageModel = os.path.join(folder_path, "age_net.caffemodel")
ageNet = cv2.dnn.readNet(ageModel, ageProto)

# Face detection model setup
faceProto = os.path.join(folder_path, "opencv_face_detector.pbtxt")
faceModel = os.path.join(folder_path, "opencv_face_detector_uint8.pb")
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# Initialize DeepSort tracker
tracker = DeepSort(max_age=140, max_cosine_distance=0.3, n_init=5, max_iou_distance=0.8)

# Check if CSV file exists, and if not, create it with headers
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


# Helper function to store tracking data in CSV
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


# Helper function to update time spent in CSV
def update_time_spent_in_csv(serial_number, new_time_spent):
    rows = []
    with csv_lock:
        if os.path.getsize(csv_file) == 0:
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


# Process each frame and apply detection, tracking, and classification
def process_frame(img, frame_number, fps):
    global serial_number, male_count, female_count

    img_copy = img.copy()
    result_img, detections, boxes = predict_objects(
        img_copy, model, classes=[0], conf=0.5
    )
    tracks = track_objects(detections, img)

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

        elapsed_time = (abs(frame_number) - track_info[track_id]["start_frame"]) / fps
        if elapsed_time > 0:
            track_info[track_id]["time_spent"] = elapsed_time

        track_info[track_id]["last_appearance_time"] = current_time
        executor.submit(
            update_time_spent_in_csv,
            track_info[track_id]["serial_number"],
            elapsed_time,
        )

        text_lines = [
            f"ID: {track_info[track_id]['serial_number']}",
            f"Gender: {track_info[track_id]['gender']}",
            f"Time: {elapsed_time:.2f}s",
            f"Age: {track_info[track_id]['age']}",
        ]
        for i, line in enumerate(text_lines):
            y_position = y1 - 10 - i * 20
            cv2.putText(
                img,
                line,
                (x1, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        current_tracks.add(track_id)

    disappeared_tracks = set(track_info.keys()) - current_tracks
    for track_id in disappeared_tracks:
        del track_info[track_id]

    return img


def predict_objects(img, chosen_model, classes=[0], conf=0.3):
    results = chosen_model.predict(img, classes=classes, conf=conf)
    detections, boxes = [], []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = (
                int(box.xyxy[0][0]),
                int(box.xyxy[0][1]),
                int(box.xyxy[0][2]),
                int(box.xyxy[0][3]),
            )
            class_id, confidence = int(box.cls[0]), box.conf[0]
            detections.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])
            boxes.append((x1, y1, x2, y2))
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                img,
                f"{class_names[class_id]}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 0, 0),
                1,
            )
    return img, detections, boxes


def track_objects(detections, img):
    tracks = tracker.update_tracks(detections, frame=img)
    return tracks


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
        return "Unknown Gender"


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video stream: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        "Processed_Output.mp4", fourcc, fps, (frame_width, frame_height)
    )

    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result_img = process_frame(frame, frame_number, fps)
        out.write(result_img)  # Write the frame to the output video
        frame_number += 1

        cv2.imshow("Inference", result_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logger.info("Video processing completed and saved to 'Processed_Output.mp4'.")


# Main execution
if __name__ == "__main__":
    process_video("/Users/ramesh/Downloads/upwork/CrowdIQ V1 MS/sample.mp4")
