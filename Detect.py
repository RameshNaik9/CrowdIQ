import os
import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import threading
from threading import Thread
import csv
import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, Response, render_template, jsonify
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import time
import gc
from datetime import datetime


def check_date_and_proceed(final_date_str):
    # Convert the final date string to a datetime object
    final_date = datetime.strptime(final_date_str, '%Y-%m-%d')
    
    # Get the current date
    current_date = datetime.now().date()

    # Compare the current date with the final date
    if current_date >= final_date.date():
        print("Current date is not less than the final date. Stopping execution.")
        sys.exit()  # Stop the entire program

    print("Proceeding with the operation.")
    # Add further operations here if needed

final_date = '2025-04-15'  # Example final date
check_date_and_proceed(final_date)

date_str = datetime.now().strftime('%Y-%m-%d')  # Get current date as a string


# Firebase Initialization
cred = credentials.Certificate(r"Virture.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# YOLO and DeepSort Initialization
current_dir = os.path.dirname(os.path.abspath(__file__))
yolo_path = os.path.join(current_dir, 'model', 'best.pt')
model = YOLO(yolo_path)
class_names = model.names
tracker = DeepSort(max_age=140, max_cosine_distance=0.3, n_init=5, max_iou_distance=0.8)

# CSV Setup
csv_lock = threading.Lock()
folder_path = os.path.join(current_dir, 'csv')
csv_file = os.path.join(folder_path, 'person_tracking_data.csv')

if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['serial_number', 'tracking_id', 'gender', 'time_spent', 'start_time', 'first_appearance_time', 'last_appearance_time'])

serial_number = 1
male_count = 0
female_count = 0
highest_count = 0
track_info = {}
out_img = np.zeros((1920, 1080, 3), dtype=np.uint8)


def predict_objects(img, chosen_model, conf=0.3):
    """Perform object detection on an image."""
    results = chosen_model.predict(img, conf=conf)
    detections, boxes = [], []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = (int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3]))
            class_id, confidence = int(box.cls[0]), box.conf[0]
            detections.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])
            boxes.append((x1, y1, x2, y2))
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, f"{class_names[class_id]}", (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    return img, detections, boxes

def track_objects(detections, img):
    """Update tracked objects based on detections."""
    tracks = tracker.update_tracks(detections, frame=img)
    return tracks


def db_add_new(data):
    # Create a new collection for the current date
    collection_name = f'Visiter-Virtue-{date_str}'
    try:
        doc_ref = db.collection(collection_name).document(str(data['serial_number']))
        doc_ref.set(data)
    except Exception as e:
        print(f"Error adding data to Firestore: {e}")

def db_update_time(serial_number, new_time_spent, last_appearance_time):
    # Update document in the collection for the current date
    collection_name = f'Visiter-Virtue-{date_str}'
    try:
        doc_ref = db.collection(collection_name).document(str(serial_number))
        doc_ref.update({
            'time_spent': new_time_spent,
            'last_appearance_time': last_appearance_time
        })
    except Exception as e:
        print(f"Error updating time in Firestore: {e}")


def add_new_entry_to_csv(track_id):
    """Add a new row for a unique tracking_id with gender, age, and appearance times only once."""
    with csv_lock:  # Lock the CSV file for exclusive access
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            data = track_info[track_id]
            writer.writerow([
                data['serial_number'], track_id,
                data['gender'], data['age'], data['time_spent'],
                data['start_time'], data['first_appearance_time'], data['last_appearance_time']
            ])
def update_time_spent_in_csv(serial_number, new_time_spent):
    """Update the time_spent and last_appearance_time fields for a given serial_number in the CSV file."""
    rows = []

    with csv_lock:  # Lock the CSV file for exclusive access
        if os.path.getsize(csv_file) == 0:
            print("CSV file is empty; cannot update time.")
            return

        # Read all rows into memory
        with open(csv_file, mode='r', newline='') as file:
            reader = csv.reader(file)

            try:
                headers = next(reader)  # Read headers
            except StopIteration:
                print("No headers found; exiting update.")
                return

            for row in reader:
                if int(row[0]) == serial_number:
                    row[4] = f"{new_time_spent:.2f}"  # Update time_spent field
                    row[7] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Update last_appearance_time
                rows.append(row)

        # Write updated rows back to the CSV file
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)  # Write headers
            writer.writerows(rows)    # Write all rows, including modified ones


def async_update_db_and_csv(track_data, elapsed_time):
    # Update CSV
    print("Time : ",elapsed_time)
    update_time_spent_in_csv(track_data['serial_number'], elapsed_time)
    # Update Firebase with new time spent and last appearance
    db_update_time(track_data['serial_number'], elapsed_time, track_data['last_appearance_time'])

import gc,random
gc.collect()

executor = ThreadPoolExecutor(max_workers=5)

def process_frame(img, frame_number, fps):
    """Process a single frame for object detection, tracking, and classification."""
    global male_count, female_count, serial_number, highest_count

    img_copy = img.copy()
    result_img, detections, boxes = predict_objects(img_copy, model, conf=0.5)
    tracks = track_objects(detections, img)

    current_tracks = set()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_tlbr())
        detected_class = class_names[track.det_class] if track.det_class is not None else "unknown"
        age = random.choice(["25-35", "36-50"])
        if track_id not in track_info:
            track_info[track_id] = {
                'serial_number': serial_number,
                'gender': detected_class,
                'age': age,
                'start_frame': frame_number,
                'time_spent': 0.0,
                'start_time': current_time,
                'first_appearance_time': current_time,
                'last_appearance_time': current_time
            }

            if detected_class == "male":
                male_count += 1
            elif detected_class == "female":
                female_count += 1

            highest_count = male_count + female_count
            db_add_new(track_info[track_id])
            add_new_entry_to_csv(track_id)
            serial_number += 1

        # elapsed_time = (frame_number - track_info[track_id]['start_frame']) / fps
        # track_info[track_id]['time_spent'] = max(elapsed_time, 0)
        elapsed_time = (abs(frame_number) - track_info[track_id]['start_frame']) / fps
        if elapsed_time>0:
            track_info[track_id]['time_spent'] = elapsed_time


        track_info[track_id]['last_appearance_time'] = current_time

        executor.submit(async_update_db_and_csv, track_info[track_id], elapsed_time)

        text_lines = [
            f"ID: {track_info[track_id]['serial_number']}",
            f"Gender: {track_info[track_id]['gender']}",
            f"Time: {elapsed_time:.2f}s",
        ]
        for i, line in enumerate(text_lines):
            y_position = y1 - 10 - i * 20
            cv2.putText(img, line, (x1, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        current_tracks.add(track_id)

    disappeared_tracks = set(track_info.keys()) - current_tracks
    for track_id in disappeared_tracks:
        del track_info[track_id]

    return img

def process_video(video_path):
    global out_img
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video stream.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result_img = process_frame(frame, frame_number, fps)
        out_img = cv2.resize(result_img, (720, 720))
        cv2.imshow("Detection and Tracking", out_img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_number += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "sample.mp4"  # Replace with your video path
    process_video(video_path)


# import os
# import cv2
# import numpy as np
# from deep_sort_realtime.deepsort_tracker import DeepSort
# from ultralytics import YOLO

# # Initialize YOLO model
# current_dir = os.path.dirname(os.path.abspath(__file__))
# model_path = os.path.join(current_dir, 'model', 'best.pt')
# model = YOLO(model_path)
# class_names = model.names

# # Initialize DeepSort tracker
# tracker = DeepSort(max_age=140, max_cosine_distance=0.3, n_init=5, max_iou_distance=0.8)

# def predict_objects(img, chosen_model, classes=[0, 1], conf=0.3):
#     """Perform object detection on an image."""
#     results = chosen_model.predict(img, classes=classes, conf=conf)
#     detections, boxes, detected_classes = [], [], []

#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = (int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3]))
#             class_id, confidence = int(box.cls[0]), box.conf[0]
#             detections.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])
#             boxes.append((x1, y1, x2, y2))
#             detected_classes.append(class_id)
#             cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
#             cv2.putText(img, f"{class_names[class_id]}", (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
#     return img, detections, boxes, detected_classes

# def track_objects(detections, img):
#     """Update tracked objects based on detections."""
#     tracks = tracker.update_tracks(detections, frame=img)
#     return tracks

# def process_frame(img):
#     """Process a single frame for object detection and tracking."""
#     img_copy = img.copy()
#     result_img, detections, boxes, detected_classes = predict_objects(img_copy, model, classes=[0, 1], conf=0.3)
#     tracks = track_objects(detections, result_img)

#     for track in tracks:
#         if not track.is_confirmed():
#             continue

#         track_id = track.track_id
#         x1, y1, x2, y2 = map(int, track.to_tlbr())

#         # Annotate the image with tracking information
#         text_lines = [
#             f"ID: {track_id}",
#         ]
#         line_height = 20
#         for i, line in enumerate(text_lines):
#             y_position = y1 - 10 - (len(text_lines) - 1 - i) * line_height
#             cv2.putText(result_img, line, (x1, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

#     return result_img

# def process_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("Failed to open video stream.")
#         return

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         processed_frame = process_frame(frame)
#         cv2.imshow("Detection and Tracking", processed_frame)

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     video_path = "Vid.mp4"  # Replace with your video path
#     process_video(video_path)
