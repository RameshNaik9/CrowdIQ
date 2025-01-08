import vlc
import ctypes
import numpy as np


# Set up global variables for the callback
width, height = 1280, 720  # Set to your stream resolution
frame_size = width * height * 4  # Assuming RGBA format

frame_buffer = None
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

