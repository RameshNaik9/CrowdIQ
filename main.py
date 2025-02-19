from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import threading
import time
import asyncio
import torch
import random
import numpy as np

# Import models and trackers
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ------------------------------------------------------------------------------
# FastAPI app setup
# ------------------------------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the main asyncio event loop for thread-safe scheduling.
main_loop = asyncio.get_event_loop()

# Dictionary to hold active stream threads and their info.
active_streams = {}


# ------------------------------------------------------------------------------
# Request Model
# ------------------------------------------------------------------------------
class StartInferenceRequest(BaseModel):
    camera_id: str
    rtsp_url: str


# ------------------------------------------------------------------------------
# WebSocket Manager to handle multiple client connections
# ------------------------------------------------------------------------------
class WebSocketManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, camera_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[camera_id] = websocket

    def disconnect(self, camera_id: str):
        if camera_id in self.active_connections:
            del self.active_connections[camera_id]

    async def send_frame(self, camera_id: str, frame: bytes):
        if camera_id in self.active_connections:
            websocket = self.active_connections[camera_id]
            try:
                await websocket.send_bytes(frame)
            except WebSocketDisconnect:
                self.disconnect(camera_id)


ws_manager = WebSocketManager()

# ------------------------------------------------------------------------------
# Model Initialization (integrated from Detect3.py)
# ------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# Initialize YOLO model for object detection.
# Adjust the model path as needed.
yolo_model_path = "model/yolo11n.pt"
yolo_model = YOLO(yolo_model_path)
yolo_model = yolo_model.to(device)

# Initialize DeepSort tracker.
tracker = DeepSort(
    max_age=140,
    max_cosine_distance=0.3,
    n_init=5,
    max_iou_distance=0.8,
)

# Initialize gender classification model.
gender_processor = AutoImageProcessor.from_pretrained(
    "rizvandwiki/gender-classification"
)
gender_model = AutoModelForImageClassification.from_pretrained(
    "rizvandwiki/gender-classification"
)

# ------------------------------------------------------------------------------
# Processing Functions
# ------------------------------------------------------------------------------


def process_frame(img, frame_number, fps):
    """
    Process a single frame:
    - Run YOLO object detection (for class 0, typically "person").
    - Track objects with DeepSort.
    - For each confirmed track, perform gender prediction,
      assign a dummy age, and annotate the frame.
    """
    # Run YOLO detection on the image.
    results = yolo_model.predict(img, classes=[0], conf=0.5)
    detections = []
    for result in results:
        for box in result.boxes:
            # Get bounding box coordinates.
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            detections.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

    # Track detected objects using DeepSort.
    tracks = tracker.update_tracks(detections, frame=img)

    # Process each confirmed track.
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_tlbr())

        # Extract region of interest for gender prediction.
        track_roi = img[y1:y2, x1:x2]
        try:
            if track_roi.size == 0:
                gender = "Unknown"
            else:
                track_roi_resized = cv2.resize(track_roi, (224, 224))
                inputs = gender_processor(images=track_roi_resized, return_tensors="pt")
                with torch.no_grad():
                    outputs = gender_model(**inputs)
                predicted_class_idx = outputs.logits.argmax(-1).item()
                gender = gender_model.config.id2label.get(
                    predicted_class_idx, "Unknown"
                )
        except Exception as e:
            gender = "Unknown"

        # Calculate elapsed time (as a simple example, using frame number).
        elapsed_time = frame_number / fps

        # Assign age randomly (as in Detect3.py demo).
        age = random.choice(["25-35", "36-50"])

        # Draw bounding box and overlay tracking info.
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"ID:{track_id} {gender} {age} {elapsed_time:.2f}s"
        cv2.putText(
            img,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return img


def process_stream(camera_id, rtsp_url):
    """
    Open the RTSP stream, process each frame using the detection/tracking
    pipeline, and stream the processed frames via WebSocket.
    """
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open RTSP stream: {rtsp_url}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 20  # fallback FPS value if not available
    frame_number = 0

    # (Optional) Uncomment the following to save processed video locally.
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # out = cv2.VideoWriter("Processed_Output.mp4", fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[INFO] Stream ended for camera_id: {camera_id}")
            break

        processed_frame = process_frame(frame, frame_number, fps)

        # (Optional) Write the processed frame to a video file.
        # out.write(processed_frame)

        # Encode the processed frame as JPEG.
        ret2, buffer = cv2.imencode(".jpg", processed_frame)
        if not ret2:
            print("[ERROR] Failed to encode frame")
            continue
        frame_bytes = buffer.tobytes()

        # Asynchronously send the frame via WebSocket.
        try:
            main_loop.call_soon_threadsafe(
                asyncio.create_task, ws_manager.send_frame(camera_id, frame_bytes)
            )
        except Exception as e:
            print(f"[ERROR] Failed to send frame: {e}")

        frame_number += 1

        # (Optional) To avoid overwhelming CPU, you may add a small sleep.
        # time.sleep(0.01)

    cap.release()
    # (Optional) If saving video, release the writer.
    # out.release()
    if camera_id in active_streams:
        del active_streams[camera_id]
    print(f"[INFO] Released RTSP stream for camera_id: {camera_id}")


# ------------------------------------------------------------------------------
# FastAPI Endpoints
# ------------------------------------------------------------------------------


@app.post("/start-inference")
async def start_inference(request: StartInferenceRequest):
    """
    Start the inference process on the provided RTSP stream.
    This launches a new thread that processes the video stream using the
    detection/tracking pipeline and streams the processed frames.
    """
    camera_id = request.camera_id
    rtsp_url = request.rtsp_url

    if camera_id in active_streams:
        raise HTTPException(
            status_code=400,
            detail=f"Inference already running for camera_id: {camera_id}",
        )

    # Check RTSP connectivity.
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        raise HTTPException(
            status_code=400, detail=f"Failed to connect to RTSP URL: {rtsp_url}"
        )
    cap.release()

    # Start processing in a separate thread.
    stream_thread = threading.Thread(
        target=process_stream, args=(camera_id, rtsp_url), daemon=True
    )
    stream_thread.start()
    active_streams[camera_id] = {"thread": stream_thread, "rtsp_url": rtsp_url}

    return {"message": f"Inference started successfully for camera_id: {camera_id}"}


@app.websocket("/ws/{camera_id}")
async def websocket_endpoint(camera_id: str, websocket: WebSocket):
    """
    WebSocket endpoint for streaming processed frames.
    Clients should connect to receive JPEG-encoded frames.
    """
    await ws_manager.connect(camera_id, websocket)
    try:
        while True:
            # Keep the connection alive.
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(camera_id)


# ------------------------------------------------------------------------------
# To run the application:
# Use the command: uvicorn main:app --reload
# ------------------------------------------------------------------------------
