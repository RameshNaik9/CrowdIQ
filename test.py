import os
import cv2
import numpy as np
import firebase_admin
from flask import Flask, Response, render_template, jsonify,request,redirect,url_for
from concurrent.futures import ThreadPoolExecutor
import gc
from utils1 import*
from utils_db import*



check_date_and_proceed(final_date)
cred = credentials.Certificate(r"Virture.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
current_dir = os.path.dirname(os.path.abspath(__file__))

# CSV Setup
folder_path = os.path.join(current_dir, 'csv')
csv_file = os.path.join(folder_path, 'person_tracking_data.csv')
create_csv(csv_file)
out_img = np.zeros((1920, 1080, 3), dtype=np.uint8)


gc.collect()

executor = ThreadPoolExecutor(max_workers=5)


def process_video(video_path):
    global out_img
    cap = cv2.VideoCapture(video_path)
    cam = Camera(video_path)

    if not cap.isOpened():
        print("Failed to open video stream.")
        return

  

    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    frame_number = 0

    while True:
        ret, frame1 = cap.read()
        frame = cam.get_frame(0.65)  # Resize frame to 65% of original size

        if not ret:
            break
        
        frame = cv2.resize(frame, (1080, 720))
        result_img = process_frame(frame, frame_number, fps,db,csv_file,executor)
        out_img = cv2.resize(result_img, (1080, 720))
        
        cv2.imshow("Detection and Tracking", out_img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_number += 1

    cap.release()
    cv2.destroyAllWindows()

# if __name__ == "__main__":
#     video_path = "rtsp://admin:Ksa11223344@192.168.1.63:554/Streaming/Channels/201"   # Replace with your video path
#     process_video(video_path)

rtsp_url = ""  # Store the RTSP URL provided by the user
app = Flask(__name__)


@app.route('/rtsp', methods=['GET', 'POST'])  # Allow both GET and POST
def submit_rtsp_data():
    global rtsp_url
    if request.method == 'POST':
        # Retrieve RTSP data from the form submission
        username = request.form['username']
        password = request.form['password']
        ip = request.form['ip']
        port = request.form['port']
        channel = request.form['channel']
        stream = request.form['stream']

        # Construct the RTSP URL
        rtsp_url = f"rtsp://{username}:{password}@{ip}:{port}/Streaming/Channels/{channel}{stream}"
        print(rtsp_url)

        # Redirect to the start page after setting the RTSP URL
        return redirect(url_for('index'))
    return render_template('rtsp.html')  # Render your form page if GET


@app.route('/')
def start_page():
    global flag_loop
    flag_loop = False
    return render_template('start_page.html')  # Initial start page

@app.route('/highest_count')
def get_highest_count():
    global male_count,female_count,highest_count
    return jsonify({
        'highest_count': highest_count,
        'male_count': male_count,
        'female_count': female_count
    })

video_thread = None  # Global variable to track the video processing thread
thread_running = False  # Global flag to indicate if a thread is already running

@app.route('/index2')
def index():
    global flag_loop, video_thread, thread_running

    flag_loop = True

    # Check if a thread is already running
    if not thread_running:
        # Start the video processing thread
        thread_running = True
        video_thread = threading.Thread(target=run_video_processing, daemon=True)

        video_thread.start()
    
    return render_template('index2.html')  # Main visitor count page

def run_video_processing():
    global thread_running
    try:
        process_video(rtsp_url)
    except Exception as e:
        print(f"Error in video processing: {e}")
    finally:
        thread_running = False

@app.route('/video')
def video_page():
    return render_template('video.html')  # Video stream page

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames_for_web(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)