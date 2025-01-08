from datetime import datetime
from deep_sort_realtime.deepsort_tracker import DeepSort

import cv2, os, time,numpy as np
from ultralytics import YOLO
import sys,random
import numpy as np
from utils_db import*

date_str = datetime.now().strftime('%Y-%m-%d')  # Get current date as a string


# YOLO and DeepSort Initialization
current_dir = os.path.dirname(os.path.abspath(__file__))
yolo_path = os.path.join(current_dir, 'model', 'yolov8_v1.pt')
model = YOLO(yolo_path)

tracker = DeepSort(max_age=140, max_cosine_distance=0.2, n_init=3, max_iou_distance=0.7)
class_names = model.names

out_img=img = np.zeros((1920, 1080, 3), dtype=np.uint8)




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

track_info = {}
serial_number = 1
male_count = 0
female_count = 0
highest_count = 0

def show_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        frame = param['frame']
        # Create a copy of the frame to overlay coordinates
        frame_with_coords = frame.copy()
        text = f"X: {x}, Y: {y}"
        cv2.putText(frame_with_coords, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Live Video', frame_with_coords)


def process_frame(img, frame_number, fps,db,csv_file,executor):
    """Process a single frame for object detection, tracking, and classification."""
    global male_count, female_count, serial_number, highest_count
    img_copy = img.copy()
    frame=img
 
    rectangle_points = [(815, 3), (1019, 54), (956, 408), (779, 297)]  # Define 4 corner points
    img_copy = draw_rectangle_with_points(img_copy, rectangle_points)


    rectangle_points = [(190, 397), (335, 301), (535, 550), (310, 717)]  # Define 4 corner points
    img_copy = draw_rectangle_with_points(img_copy, rectangle_points)

    result_img, detections, boxes = predict_objects(img_copy, model, conf=0.5)
    img=result_img
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
            db_add_new(db,track_info[track_id],date_str)
            add_new_entry_to_csv(csv_file,track_info,track_id)
            serial_number += 1

        # elapsed_time = (frame_number - track_info[track_id]['start_frame']) / fps
        # track_info[track_id]['time_spent'] = max(elapsed_time, 0)
        elapsed_time = (abs(frame_number) - track_info[track_id]['start_frame']) / fps
        if elapsed_time>0:
            track_info[track_id]['time_spent'] = elapsed_time
        track_info[track_id]['last_appearance_time'] = current_time
        executor.submit(async_update_db_and_csv,db, csv_file,track_info[track_id], elapsed_time,date_str)

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
    global out_img
    out_img=img
    return img


def draw_rectangle_with_points(img, points, color=(0, 0, 255), thickness=3):
   
    if len(points) != 4:
        raise ValueError("Exactly 4 points are required to draw a rectangle.")

    # Convert points to integer if they are not already
    points = np.array(points, dtype=np.int32)

    # Fill the polygon (rectangle) with the specified color
    cv2.fillPoly(img, [points], color)

    return img




import cv2
import time
import multiprocessing as mp

class Camera():
    
    def __init__(self,rtsp_url):        
        #load pipe for data transmittion to the process
        self.parent_conn, child_conn = mp.Pipe()
        #load process
        self.p = mp.Process(target=self.update, args=(child_conn,rtsp_url))        
        #start process
        self.p.daemon = True
        self.p.start()
        
    def end(self):
        #send closure request to process
        
        self.parent_conn.send(2)
        
    def update(self,conn,rtsp_url):
        #load cam into seperate process
        
        print("Cam Loading...")
        cap = cv2.VideoCapture(rtsp_url,cv2.CAP_FFMPEG)   
        print("Cam Loaded...")
        run = True
        
        while run:
            
            #grab frames from the buffer
            cap.grab()
            
            #recieve input data
            rec_dat = conn.recv()
            
            
            if rec_dat == 1:
                #if frame requested
                ret,frame = cap.read()
                conn.send(frame)
                
            elif rec_dat ==2:
                #if close requested
                cap.release()
                run = False
                
        print("Camera Connection Closed")        
        conn.close()
    
    def get_frame(self,resize=None):
        ###used to grab frames from the cam connection process
        
        ##[resize] param : % of size reduction or increase i.e 0.65 for 35% reduction  or 1.5 for a 50% increase
             
        #send request
        self.parent_conn.send(1)
        frame = self.parent_conn.recv()
        
        #reset request 
        self.parent_conn.send(0)
        
        #resize if needed
        if resize == None:            
            return frame
        else:
            return self.rescale_frame(frame,resize)
        
    def rescale_frame(self,frame, percent=65):
        
        return cv2.resize(frame,None,fx=percent,fy=percent) 







  # cv2.namedWindow('Detection and Tracking')

    # # Prepare the parameter dictionary for the callback
    # param = {'frame': None}

    # # Attach the mouse callback
    # cv2.setMouseCallback('Detection and Tracking', show_coordinates, param)


            # param['frame'] = frame in while True: