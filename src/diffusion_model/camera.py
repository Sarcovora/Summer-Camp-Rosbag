import cv2
import numpy as np
import pyrealsense2 as rs
import rospy
from matplotlib import pyplot as plt
from pynput import keyboard


class Camera:
    """Class to interface with Intel RealSense camera and observe image observations."""

    def __init__(self, IMG_X=640, IMG_Y=480):
        """Initialize RealSense camera pipeline to store RGB images with dimensions (IMG_X, IMG_Y)."""
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, IMG_X, IMG_Y, rs.format.bgr8, 30)
        self.pipeline.start(self.config)

    def get_frame(self):
        """Capture a frame and return it as a NumPy array with dimensions (IMG_X, IMG_Y, 3)"""
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        return color_image_rgb
    
    def get_depth_frame(self):
        """Capture a depth frame and return it as a NumPy array with dimensions (IMG_X, IMG_Y)"""
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        return depth_image

    def close(self):
        """Shuts down the RealSense camera pipeline."""
        self.pipeline.stop()


if __name__ == "__main__":
    camera = Camera()
    frame = camera.get_frame()
    print(frame)

    depth_frame = camera.get_depth_frame()
    print(depth_frame)