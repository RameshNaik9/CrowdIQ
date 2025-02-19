from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.websockets import WebSocketDisconnect
from pydantic import BaseModel
import cv2
import threading
import time
import json
import asyncio
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # Allows all origins. Replace "*" with a list of origins in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the main event loop (assuming uvicorn runs the app in the main thread)
main_loop = asyncio.get_event_loop()

# Dictionary to hold active RTSP streams
active_streams = {}


# Pydantic model for request payload
class StartInferenceRequest(BaseModel):
    camera_id: str
    rtsp_url: str


# WebSocket manager for handling multiple clients
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


# WebSocket manager instance
ws_manager = WebSocketManager()


@app.post("/start-inference")
async def start_inference(request: StartInferenceRequest):
    """
    Start video inference for a given RTSP URL.
    """
    camera_id = request.camera_id
    rtsp_url = request.rtsp_url

    # Check if the stream is already active
    if camera_id in active_streams:
        raise HTTPException(
            status_code=400,
            detail=f"Inference already running for camera_id: {camera_id}",
        )

    # Try to connect to the RTSP stream
    try:
        cap = cv2.VideoCapture(rtsp_url)

        if not cap.isOpened():
            raise HTTPException(
                status_code=400, detail=f"Failed to connect to RTSP URL: {rtsp_url}"
            )

        # Start a new thread to process the stream
        stream_thread = threading.Thread(
            target=process_stream, args=(camera_id, rtsp_url)
        )
        stream_thread.start()

        # Store the stream handle
        active_streams[camera_id] = {"thread": stream_thread, "rtsp_url": rtsp_url}

        return {"message": f"Inference started successfully for camera_id: {camera_id}"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error starting inference: {str(e)}"
        )


@app.websocket("/ws/{camera_id}")
async def websocket_endpoint(camera_id: str, websocket: WebSocket):
    """
    WebSocket endpoint for streaming processed frames to the frontend.
    """
    await ws_manager.connect(camera_id, websocket)
    try:
        while True:
            # Keep the connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(camera_id)


def process_stream(camera_id, rtsp_url):
    """
    Process the RTSP stream for inference (adds a red overlay for demonstration).
    """
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open RTSP stream: {rtsp_url}")
        return

    print(f"[INFO] Processing RTSP stream for camera_id: {camera_id}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[INFO] RTSP stream ended for camera_id: {camera_id}")
            break

        # Add a red transparent overlay to the frame (demo processing)
        overlay = frame.copy()
        overlay[:, :, 2] = 255  # Maximize the red channel
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

        # Encode the frame as JPEG
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        # Send processed frame to the frontend by scheduling the task on the main event loop
        try:
            main_loop.call_soon_threadsafe(
                asyncio.create_task, ws_manager.send_frame(camera_id, frame_bytes)
            )
        except Exception as e:
            print(f"[ERROR] Failed to send frame: {str(e)}")

        # Generate and log dummy metadata
        metadata = {
            "camera_id": camera_id,
            "timestamp": time.time(),
            "person_count": 5,  # Dummy value
            "detected_objects": [
                {"type": "person", "confidence": 0.98},
                {"type": "person", "confidence": 0.95},
            ],
        }
        # print(f"[INFO] Metadata: {json.dumps(metadata)}")

        # Terminate the loop if the stream is no longer active
        if camera_id not in active_streams:
            print(f"[INFO] Stopping inference for camera_id: {camera_id}")
            break

    # Release the stream
    cap.release()
    print(f"[INFO] Released RTSP stream for camera_id: {camera_id}")

    # Remove from active streams
    if camera_id in active_streams:
        del active_streams[camera_id]


#  uvicorn main:app --reload