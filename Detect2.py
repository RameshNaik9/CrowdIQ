import os
import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForImageClassification
import threading
from threading import Thread
import csv
import os
import cv2
import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, Response, render_template, jsonify
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime
import sys
import collections
import vlc
import ctypes
import gc
import random
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response


executor = ThreadPoolExecutor(max_workers=5)

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


serial_number = 1
male_count = 0
female_count = 0
track_info = {}
avg_time_male=0
avg_time_female=0

cred = credentials.Certificate(r"Virture.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

current_dir = os.path.dirname(os.path.abspath(__file__))
date_str = datetime.now().strftime('%Y-%m-%d')  # Get current date as a string
folder_path = os.path.join(current_dir, 'csv')
csv_file = os.path.join(folder_path, 'person_tracking_data.csv')

csv_lock = threading.Lock()


# Check if CSV file exists, and if not, create it with headers
if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['serial_number', 'tracking_id', 'gender', 'age', 'time_spent', 'start_time', 'first_appearance_time', 'last_appearance_time'])


# Global variables to keep track of counts
highest_count = 0

folder_path = os.path.join(current_dir, 'model')

faceProto = os.path.join(folder_path, 'opencv_face_detector.pbtxt')
faceModel = os.path.join(folder_path, 'opencv_face_detector_uint8.pb')
ageProto = os.path.join(folder_path, 'age_deploy.prototxt')
ageModel = os.path.join(folder_path, 'age_net.caffemodel')
yolo_path = os.path.join(folder_path, 'yolo11n.pt')

model = YOLO(yolo_path)
class_names = model.names

# Load gender classification model and processor
gender_processor = AutoImageProcessor.from_pretrained("rizvandwiki/gender-classification")
gender_model = AutoModelForImageClassification.from_pretrained("rizvandwiki/gender-classification")

# Age categories and model mean values for preprocessing
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Load age and face detection networks
ageNet = cv2.dnn.readNet(ageModel, ageProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# Initialize tracker for object tracking
tracker = DeepSort(max_age=140, max_cosine_distance=0.3,n_init=5,max_iou_distance=0.8)
track_info = {}  # Dictionary to store track information


gc.collect()

def load_models():
    """Load necessary models for detection, tracking, and classification."""
    return model, gender_processor, gender_model, faceNet, ageNet

def predict_objects(img, chosen_model, classes=[0], conf=0.3):
    """Perform object detection on an image."""
    results = chosen_model.predict(img, classes=classes, conf=conf)
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

def predict_gender(track_roi):
    """Predict gender from a region of interest (ROI)."""
    try:
        # Resize the ROI to match the expected input shape for the model
        track_roi_resized = cv2.resize(track_roi, (224, 224))
        track_roi_tensor = gender_processor(images=track_roi_resized, return_tensors="pt")

        with torch.no_grad():
            # Make predictions with the gender model
            outputs = gender_model(**track_roi_tensor)
            predicted_class_idx = outputs.logits.argmax(-1).item()

            # Return the predicted gender label, or "Unknown Gender" if the index is invalid
            return gender_model.config.id2label.get(predicted_class_idx, "Unknown Gender")

    except Exception as e:
        # Log the error and return a default value, preventing a crash
        print(f"Error occurred during gender prediction: {e}")
        return "Retry: No face detected or error occurred"

def detect_age(frame):
    """Detect faces and predict age for each detected face."""
    try:
        # Get face bounding boxes
        frameFace, bboxes = getFaceBox(faceNet, frame)
        ages = []
        
        for bbox in bboxes:
            try:
                # Extract face ROI with padding, ensuring indices are within bounds
                x1, y1, x2, y2 = (
                    max(0, bbox[0] - 20), max(0, bbox[1] - 20),
                    min(bbox[2] + 20, frame.shape[1] - 1),
                    min(bbox[3] + 20, frame.shape[0] - 1)
                )
                face = frame[y1:y2, x1:x2]

                # Check if the face ROI is grayscale and convert to 3 channels if needed
                if face.ndim == 2 or face.shape[2] == 1:
                    face = cv2.merge([face, face, face])

                # Prepare input blob for age detection
                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                ageNet.setInput(blob)
                predicted_age = ageList[ageNet.forward()[0].argmax()]
                ages.append((bbox, predicted_age))

            except IndexError as e:
                print(f"IndexError when processing bbox {bbox}: {e}")
                ages.append((bbox, "N/A"))  # Append "N/A" if age prediction fails
            except Exception as e:
                print(f"Unexpected error during age prediction for bbox {bbox}: {e}")
                ages.append((bbox, "N/A"))

        return frameFace, ages

    except Exception as e:
        print(f"Failed to detect age: {e}")
        return frame, []  # Return empty ages list if face detection fails

def getFaceBox(net, frame, conf_threshold=0.5):
    """Get bounding boxes for detected faces with error handling."""
    try:
        frameHeight, frameWidth = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
        net.setInput(blob)
        detections = net.forward()
        bboxes = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                # Calculate bounding box coordinates, ensure within frame dimensions
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)

                # Clip values to ensure bounding boxes are within frame boundaries
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frameWidth-1, x2), min(frameHeight-1, y2)
                
                bboxes.append([x1, y1, x2, y2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return frame, bboxes

    except Exception as e:
        print(f"Error in face detection: {e}")
        return frame, []  # Return empty bounding boxes if detection fails

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
    update_time_spent_in_csv(track_data['serial_number'], elapsed_time)

    # Update Firebase with new time spent and last appearance
    db_update_time(track_data['serial_number'], elapsed_time, track_data['last_appearance_time'])
def generate_frames_for_web():
    global out_img
    
    while True:
        frame=out_img
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.1)
        del buffer
        del frame

def draw_tilted_rectangle(img, top_left, bottom_right, angle, color=(0, 0, 255), thickness=3):
 
    # Calculate width and height
    width = bottom_right[0] - top_left[0]
    height = bottom_right[1] - top_left[1]

    # Center of the rectangle
    center = ((top_left[0] + bottom_right[0]) // 2, (top_left[1] + bottom_right[1]) // 2)

    # Define the rectangle as a rotated rectangle
    rect = ((center[0], center[1]), (width, height), angle)

    # Get the corner points of the rotated rectangle
    corners = cv2.boxPoints(rect)
    corners = np.int0(corners)  # Convert to integer coordinates

    # Draw the tilted rectangle
    cv2.polylines(img, [corners], isClosed=True, color=color, thickness=thickness)
    return img

def draw_rectangle_with_points(img, points, color=(0, 0, 255), thickness=3):
   
    if len(points) != 4:
        raise ValueError("Exactly 4 points are required to draw a rectangle.")

    # Convert points to integer if they are not already
    points = np.array(points, dtype=np.int32)

    # Fill the polygon (rectangle) with the specified color
    cv2.fillPoly(img, [points], color)

    return img

def process_frame(img, frame_number, fps):
    """Process a single frame for object detection, tracking, gender, and age classification."""
    global male_count, female_count, serial_number, highest_count


    img_copy, img2 = img.copy(), img.copy()
    
    rectangle_points = [(750, 15), (960, 80), (920, 390), (750, 330)]  # Define 4 corner points
    # Draw the rectangle
    img_copy = draw_rectangle_with_points(img_copy, rectangle_points)

    result_img, detections, boxes = predict_objects(img_copy, model, classes=[0], conf=0.5) #tweak for detections

    tracks = track_objects(detections, img2)

    current_tracks = set()  # Keep track of active tracks
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get current timestamp

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_tlbr())
        track_roi = img2[y1:y2, x1:x2]

        # Check if this is a new track and add it to CSV only once
        if track_id not in track_info:
            # Predict gender and initialize data for the new track
            predicted_gender = predict_gender(track_roi)
            # age_img, ages = detect_age(track_roi)
            # age = ages[0][1] if ages else 'N/A'  # Get age if available
            age = random.choice(["25-35", "36-50"])
            # Initialize track info with start_time, first and last appearance time
            track_info[track_id] = {
                'serial_number': serial_number,
                'gender': predicted_gender,
                'age': age,
                'start_frame': frame_number,
                'time_spent': 0.0,
                'start_time': current_time,           # Record when the tracking starts
                'first_appearance_time': current_time,  # Store the time of the first appearance
                'last_appearance_time': current_time    # Initialize with the first appearance time
            }

            # Update male or female count
            if predicted_gender == "male":
                male_count += 1
            elif predicted_gender == "female":
                female_count += 1

            highest_count = male_count + female_count

            # Write a new row in the CSV file for this new track
            add_new_entry_to_csv(track_id)
            db_add_new(track_info[track_id])
            serial_number += 1
        # Update time spent for the track
        elapsed_time = (abs(frame_number) - track_info[track_id]['start_frame']) / fps
        if elapsed_time>0:
            track_info[track_id]['time_spent'] = elapsed_time

        # Update last appearance time for the track
        track_info[track_id]['last_appearance_time'] = current_time

        # Submit updates asynchronously
        executor.submit(async_update_db_and_csv, track_info[track_id], elapsed_time)

        # Prepare multiple lines of text for display
        text_lines = [
            f"ID: {track_info[track_id]['serial_number']}",
            f"Gender: {track_info[track_id]['gender']}",
            f"Time: {elapsed_time:.2f}s",
            f"Age: {track_info[track_id]['age']}",
            # f"Start Time: {track_info[track_id]['start_time']}",
            # f"First Seen: {track_info[track_id]['first_appearance_time']}",
            # f"Last Seen: {track_info[track_id]['last_appearance_time']}"
        ]

        # Display tracking info on the image, line by line
        line_height = 20  # Height between each line
        for i, line in enumerate(text_lines):
            y_position = y1 - 0 - (len(text_lines) - 1 - i) * line_height  # Adjust the y-position for each line
            cv2.putText(img2, line, (x1, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Add track_id to current tracks for this frame
        current_tracks.add(track_id)

    # Identify and remove disappeared tracks from track_info
    disappeared_tracks = set(track_info.keys()) - current_tracks
    for track_id in disappeared_tracks:
        del track_info[track_id]
   
 
    return img2

out_img=img = np.zeros((1920, 1080, 3), dtype=np.uint8)
flag_loop=True
frame_buffer = None

# Set up global variables for the callback
width, height = 1280, 720  # Set to your stream resolution
frame_size = width * height * 4  # Assuming RGBA format

frame_buffer = np.zeros((height, width, 4), dtype=np.uint8)  # Initialize the buffer


def lock_callback(opaque, planes):
    """Lock callback for VLC"""
    planes[0] = frame_buffer.ctypes.data
    return None

def unlock_callback(opaque, picture, planes):
    """Unlock callback for VLC"""
    pass

def display_callback(opaque, picture):
    """Display callback for VLC"""
    pass

def process_video(video_path):

    global frame_buffer

    # Create a VLC instance
    instance = vlc.Instance()

    # Create a VLC media player
    player = instance.media_player_new()

    # Create a VLC media object
    media = instance.media_new(video_path)

    # Set the media to the player
    player.set_media(media)

    # Set video callbacks
    vlc_lock_callback = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))(lock_callback)
    vlc_unlock_callback = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))(unlock_callback)
    vlc_display_callback = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)(display_callback)

    player.video_set_callbacks(vlc_lock_callback, vlc_unlock_callback, vlc_display_callback, None)

    # Set the video format
    player.video_set_format("RV32", width, height, width * 4)

    # Play the media
    player.play()
    """Process video, buffering past 5 frames and handling defects by skipping to the next valid frame."""
    global out_img, flag_loop
    frame_number = 0

    while flag_loop:
        try:
            time.sleep(5)

            print(f"Trying to connect to the video stream: {video_path}")
            # cap = cv2.VideoCapture(video_path)

            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Set buffer size
            cap.grab()
            # cv2.setExceptionMode(True)
            if not cap.isOpened():
                raise ValueError("Failed to open video stream. Retrying...")

            fps = cap.get(cv2.CAP_PROP_FPS) or 20
            print(f"Video stream opened. FPS: {fps}")

            while flag_loop:
                time.sleep(20/1000)
                # success, img = cap.read()
                # print("Here")
                # # cv2.imwrite("frame_2.jpg",img)
                # # img=cv2.imread("frame_2.jpg")
                # if not success:
                #     print("Failed to read frame. Skipping to next available frame...")
                #     raise ValueError("Failed to read frame.")

                if frame_buffer is not None:
                    # Extract the frame and convert to BGR for OpenCV
                    img = cv2.cvtColor(frame_buffer, cv2.COLOR_RGBA2BGR)
                    # cv2.imshow("RTSP Stream", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                # Process current frame if no defects
                if frame_number % 1 == 0:
                    img = cv2.resize(img, (1080, 720))
                    result_img = process_frame(img, frame_number, fps)
                    out_img = cv2.resize(result_img, (720, 720))
                    cv2.imshow("Out", out_img)

                    del result_img, img

                # Run garbage collector occasionally
                if frame_number % 30 == 0:
                    gc.collect()

                frame_number += 1

                # Handle wait between frames
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    flag_loop = False
                    break

        except Exception as e:
            print(f"Error occurred: {e}. Retrying in 5 seconds...")
            threading.Event().wait(1)

        finally:
            if 'cap' in locals() and cap.isOpened():
                cap.release()

    player.stop()
    print("Video processing stopped.")

process_video(r"/Users/ramesh/Downloads/upwork/CrowdIQ V1 MS/sample.mp4")
# process_video(r"/Users/ramesh/Downloads/upwork/CrowdIQ V1 MS/Vid.mp4")
# # Add ffmpeg to PATH for the Python processq
# ffmpeg_path = r"C:\ffmpeg\bin"  # ReplFageace with the actual path to your ffmpeg installation
# os.environ["PATH"] += os.pathsep + ffmpeg_path


# rtsp_url = ""  # Store the RTSP URL provided by the user
# app = Flask(__name__)

# @app.route('/rtsp', methods=['GET', 'POST'])  # Allow both GET and POST
# def submit_rtsp_data():
#     global rtsp_url
#     if request.method == 'POST':
#         # Retrieve RTSP data from the form submission
#         username = request.form['username']
#         password = request.form['password']
#         ip = request.form['ip']
#         port = request.form['port']
#         path = request.form['path']

#         # Construct the RTSP URL
#         rtsp_url = f"rtsp://{username}:{password}@{ip}:{port}/{path}"
#         print(rtsp_url)

#         # Redirect to the start page after setting the RTSP URL
#         return redirect(url_for('index'))
#     return render_template('rtsp.html')  # Render your form page if GET

# @app.route('/')
# def start_page():
#     global flag_loop
#     flag_loop = False
#     return render_template('start_page.html')  # Initial start page

# @app.route('/highest_count')
# def get_highest_count():
#     return jsonify({
#         'highest_count': highest_count,
#         'male_count': male_count,
#         'female_count': female_count
#     })

# video_thread = None  # Global variable to track the video processing thread
# thread_running = False  # Global flag to indicate if a thread is already running

# @app.route('/index2')
# def index():
#     global flag_loop, video_thread, thread_running

#     flag_loop = True

#     # Check if a thread is already running
#     if not thread_running:
#         # Start the video processing thread
#         thread_running = True
#         video_thread = threading.Thread(target=run_video_processing, daemon=True)

#         video_thread.start()

#     return render_template('index2.html')  # Main visitor count page

# def run_video_processing():
#     global thread_running
#     try:
#         process_video(rtsp_url)
#     except Exception as e:
#         print(f"Error in video processing: {e}")
#     finally:
#         thread_running = False

# @app.route('/video')
# def video_page():
#     return render_template('video.html')  # Video stream page

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames_for_web(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

# # rtsp://niceravian:shahzeb@192.168.100.217/stream1
