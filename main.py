"""
File: main.py
Description: FastAPI backend with routes for /check-stream, /start-inference, /stop-inference,
and a WebSocket for streaming processed frames.
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import threading
import time
import asyncio
import torch
import random
import numpy as np
import queue
import websockets
import json
import logging
import aiohttp
from datetime import datetime

# Import models and trackers
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from transformers import AutoImageProcessor, AutoModelForImageClassification


NODEJS_API_URL = "http://localhost:8080/api/v1/inference/trigger"
NODEJS_WS_URL = "ws://localhost:8080"


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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FastAPI")


# Get the main asyncio event loop for thread-safe scheduling.
main_loop = asyncio.get_event_loop()

# Dictionary to hold active stream threads and their info.
active_streams = {}

# Global dictionary to hold stop flags for each camera.
stop_flags = {}

# Global flag for buffered streaming optimization.
USE_BUFFERED_STREAMING = True
# Dictionary to hold a per-camera buffer queue (size 1) for frames.
frame_buffers = {}


# ------------------------------------------------------------------------------
# Request Models
# ------------------------------------------------------------------------------
class StartInferenceRequest(BaseModel):
    user_id: str
    camera_id: str
    rtsp_url: str


class StopInferenceRequest(BaseModel):
    camera_id: str


# ------------------------------------------------------------------------------
# WebSocket Manager to handle multiple client connections
# ------------------------------------------------------------------------------
class WebSocketManager:
    """
    Manages active WebSocket connections.
    Allows multiple clients to subscribe to the same `camera_id` stream.
    """

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, camera_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[camera_id] = websocket

    def disconnect(self, camera_id: str):
        if camera_id in self.active_connections:
            del self.active_connections[camera_id]

    async def send_frame(self, camera_id: str, frame: bytes):
        """Send a JPEG-encoded frame to the WebSocket corresponding to `camera_id`."""
        if camera_id in self.active_connections:
            websocket = self.active_connections[camera_id]
            try:
                await websocket.send_bytes(frame)
            except WebSocketDisconnect:
                self.disconnect(camera_id)

    async def send_text(self, camera_id: str, message: str):
        """Send a text message (e.g., 'inference_stopped') to the WebSocket."""
        if camera_id in self.active_connections:
            websocket = self.active_connections[camera_id]
            try:
                await websocket.send_text(message)
            except WebSocketDisconnect:
                self.disconnect(camera_id)


ws_manager = WebSocketManager()

# ------------------------------------------------------------------------------
# Model Initialization
# ------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# Initialize YOLO model for object detection.
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
# Utility/Processing Functions
# ------------------------------------------------------------------------------
def process_frame(img, frame_number, fps, camera_id, user_id):
    """
    Process a single frame:
    - Run YOLO object detection (for class 0, typically "person").
    - Track objects with DeepSort.
    - For each confirmed track, perform gender prediction,
      assign a dummy age, and annotate the frame.
    """
    results = yolo_model.predict(img, classes=[0], conf=0.5)
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            detections.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

    # Update DeepSort tracker
    tracks = tracker.update_tracks(detections, frame=img)

    # Generate current date in UTC
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Annotate each confirmed track
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_tlbr())
        track_roi = img[y1:y2, x1:x2]

        # Gender classification
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

        # Example: random "age" & annotation
        elapsed_time = frame_number / fps
        age = random.choice(["25-35", "36-50"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"ID:{track_id} {gender} {age} {elapsed_time:.2f}s"
        cv2.putText(
            img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

        # Prepare result payload
        result_payload = {
            "userId": user_id,
            "cameraId": camera_id,
            "date": current_date,
            "track_id": str(track_id),
            "gender": gender,
            "age": age,
            "time_spent": int(elapsed_time),
            "first_appearance": str(datetime.now()),
            "last_appearance": str(datetime.now()),
        }

        # Send inference result to Node.js WebSocket
        asyncio.run_coroutine_threadsafe(
            send_inference_result_to_nodejs(result_payload), main_loop
        )
    return img


async def stream_frames(camera_id: str):
    """
    Asynchronous task to continuously get the latest frame from the
    buffer and send it over the WebSocket.
    """
    while camera_id in frame_buffers:
        try:
            # Try to get the latest frame without blocking.
            frame_bytes = frame_buffers[camera_id].get(block=False)
        except queue.Empty:
            await asyncio.sleep(0.01)
            continue

        await ws_manager.send_frame(camera_id, frame_bytes)
        # A very short sleep can help with cooperative multitasking.
        await asyncio.sleep(0.001)


def process_stream(camera_id, rtsp_url, user_id):
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

    # Initialize the frame buffer for this camera if using buffered streaming.
    if USE_BUFFERED_STREAMING:
        frame_buffers[camera_id] = queue.Queue(maxsize=1)
        # Start an asynchronous task to stream frames from the buffer.
        main_loop.call_soon_threadsafe(asyncio.create_task, stream_frames(camera_id))

    # Initialize stop flag for this camera
    stop_flags[camera_id] = False

    while True:
        # Check if a stop has been requested
        if stop_flags.get(camera_id):
            print(f"[INFO] Stop flag received for camera_id: {camera_id}")
            break

        ret, frame = cap.read()
        if not ret:
            print(f"[INFO] Stream ended for camera_id: {camera_id}")
            break

        # Pass user_id and camera_id to process_frame
        processed_frame = process_frame(frame, frame_number, fps, camera_id, user_id)

        # (Optional) Write the processed frame to a video file.
        # out.write(processed_frame)        # Encode the processed frame as JPEG
        success, buffer = cv2.imencode(".jpg", processed_frame)
        if not success:
            print("[ERROR] Failed to encode frame")
            continue
        frame_bytes = buffer.tobytes()

        if USE_BUFFERED_STREAMING:
            # Place the frame in the buffer. If full, drop the old frame.
            try:
                frame_buffers[camera_id].put(frame_bytes, block=False)
            except queue.Full:
                # Remove the oldest frame
                try:
                    frame_buffers[camera_id].get_nowait()
                except queue.Empty:
                    pass
                frame_buffers[camera_id].put(frame_bytes, block=False)
        else:
            # Send directly (not recommended if concurrency is high)
            try:
                main_loop.call_soon_threadsafe(
                    asyncio.create_task, ws_manager.send_frame(camera_id, frame_bytes)
                )
            except Exception as e:
                print(f"[ERROR] Failed to send frame: {e}")

        frame_number += 1

        # (Optional) To avoid overwhelming the CPU, you may add a small sleep.
        # time.sleep(0.01)

    cap.release()
    # (Optional) If saving video, release the writer.
    # out.release()
    if camera_id in active_streams:
        del active_streams[camera_id]
    # Clean up the frame buffer if it exists.
    if camera_id in frame_buffers:
        del frame_buffers[camera_id]
    if camera_id in stop_flags:
        del stop_flags[camera_id]
    print(f"[INFO] Released RTSP stream for camera_id: {camera_id}")

    # Notify connected WebSocket clients that inference has stopped
    asyncio.run_coroutine_threadsafe(
        ws_manager.send_text(camera_id, "inference_stopped"), main_loop
    )


async def notify_nodejs_inference_started(camera_id, rtsp_url):
    """Notify Node.js backend that inference has started."""
    async with aiohttp.ClientSession() as session:
        try:
            payload = {"cameraId": camera_id, "rtspUrl": rtsp_url, "status": "started"}
            async with session.post(NODEJS_API_URL, json=payload) as response:
                if response.status == 200:
                    logger.info(
                        f"Node.js notified successfully for camera {camera_id}."
                    )
                else:
                    logger.error(f"Failed to notify Node.js. Status: {response.status}")
        except Exception as e:
            logger.error(f"Error notifying Node.js: {e}")


async def notify_nodejs_inference_stopped(camera_id):
    """Notify Node.js backend that inference has stopped."""
    async with aiohttp.ClientSession() as session:
        try:
            payload = {"cameraId": camera_id, "status": "stopped"}
            async with session.post(NODEJS_API_URL, json=payload) as response:
                if response.status == 200:
                    logger.info(f"Node.js notified for camera {camera_id} stop.")
        except Exception as e:
            logger.error(f"Error notifying Node.js about stop: {e}")


async def send_inference_result_to_nodejs(result):
    """Send detected inference result to Node.js via WebSocket."""
    try:
        async with websockets.connect(f"{NODEJS_WS_URL}/ws") as websocket:
            await websocket.send(json.dumps(result))
            response = await websocket.recv()
            logger.info(f"Node.js Response: {response}")
    except Exception as e:
        logger.error(f"Failed to send inference result to Node.js: {e}")


# ------------------------------------------------------------------------------
# New Endpoint: Check RTSP Stream WITHOUT starting inference
# ------------------------------------------------------------------------------
@app.get("/check-stream")
def check_stream(rtsp_url: str = Query(..., description="RTSP URL to verify")):
    """
    Quickly checks if the RTSP URL can be opened.
    Does NOT start the inference thread—just opens briefly and closes.
    """
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Unable to open RTSP stream.")
    # Try to read a single frame
    ret, _ = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(
            status_code=400, detail="RTSP stream opened but no frames read."
        )
    return {"message": "RTSP stream is valid and reachable."}


# ------------------------------------------------------------------------------
# New route: /inference-status
# ------------------------------------------------------------------------------
@app.get("/inference-status")
def inference_status(camera_id: str):
    """
    Returns whether the camera_id is currently in active_streams (meaning it is running).
    """
    return {"active": camera_id in active_streams}


# ------------------------------------------------------------------------------
# Start/Stop Inference
# ------------------------------------------------------------------------------
@app.post("/start-inference")
async def start_inference(request: StartInferenceRequest):
    """
    Start the inference process on the provided RTSP stream.
    This launches a new thread that processes the video stream using the
    detection/tracking pipeline and streams the processed frames.
    """
    user_id = request.user_id
    camera_id = request.camera_id
    rtsp_url = request.rtsp_url

    if camera_id in active_streams:
        raise HTTPException(
            400, f"Inference already running for camera_id: {camera_id}"
        )

    # Notify Node.js when inference starts
    asyncio.create_task(notify_nodejs_inference_started(camera_id, rtsp_url))

    # Quick check to ensure RTSP is valid
    test_cap = cv2.VideoCapture(rtsp_url)
    if not test_cap.isOpened():
        raise HTTPException(400, f"Failed to connect to RTSP URL: {rtsp_url}")
    test_cap.release()

    # Start the processing thread with user_id and camera_id
    stream_thread = threading.Thread(
        target=process_stream, args=(camera_id, rtsp_url, user_id), daemon=True
    )
    stream_thread.start()
    active_streams[camera_id] = {"thread": stream_thread, "rtsp_url": rtsp_url}

    return {
        "message": f"Inference started for camera_id: {camera_id}, user_id: {user_id}"
    }


@app.post("/stop-inference")
async def stop_inference(request: StopInferenceRequest):
    """
    Stop the inference process for the provided camera_id.
    Sets the stop flag to gracefully exit the processing thread.
    """
    camera_id = request.camera_id
    if camera_id not in active_streams:
        raise HTTPException(
            400, f"No active inference running for camera_id: {camera_id}"
        )
    stop_flags[camera_id] = True
    await notify_nodejs_inference_stopped(camera_id)
    return {"message": f"Inference stopped successfully for camera_id: {camera_id}"}

# ------------------------------------------------------------------------------
# WebSocket Endpoint
# ------------------------------------------------------------------------------
@app.websocket("/ws/{camera_id}")
async def websocket_endpoint(camera_id: str, websocket: WebSocket):
    """
    WebSocket endpoint for streaming processed frames.
    Clients should connect to receive JPEG-encoded frames.
    """
    await ws_manager.connect(camera_id, websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep the connection alive
    except WebSocketDisconnect:
        ws_manager.disconnect(camera_id)


# ------------------------------------------------------------------------------
# To run the application:
# Use the command: uvicorn main:app --reload
# ------------------------------------------------------------------------------
