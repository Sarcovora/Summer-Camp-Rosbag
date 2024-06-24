#!/usr/bin/env python3

import os
import signal
import subprocess
import time
import datetime
import threading

# Global variable to control the recording process
recording = True

def signal_handler(sig, frame):
    global recording
    print("\nStopping recording...")
    recording = False

def record_rosbag(now):
    global recording
    process = subprocess.Popen([
        'rosbag', 'record', '-O', f'SLAM_{now}.bag', '-b', '0',
        '/camera/aligned_depth_to_color/camera_info',
        '/camera/aligned_depth_to_color/image_raw/compressedDepth',
        '/camera/color/camera_info',
        '/camera/color/image_raw/compressed',
        '/camera/imu',
        '/camera/imu_info',
        '/tf_static',
        '/tf'
    ])
    while recording:
        time.sleep(1)
    process.terminate()
    process.wait()

def main():
    signal.signal(signal.SIGINT, signal_handler)

    # Set the current time for the filename
    now = datetime.datetime.now().strftime("%F__%H_%M_%S")

    # Set ROS parameters
    subprocess.run(['rosparam', 'set', 'use_sim_time', 'false'])

    # Launch the RealSense camera node
    subprocess.Popen(['roslaunch', 'realsense2_camera', 'opensource_tracking.launch'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    input("Press Enter to start recording\n")
    print("Press 's' and Enter to stop recording")

    # Start recording in a separate thread
    record_thread = threading.Thread(target=record_rosbag, args=(now,))
    record_thread.start()

    while recording:
        if input() == 's':
            signal_handler(None, None)

    # Ensure the recording thread has finished
    record_thread.join()

    # Prompt the user afterwards
    input("Recording stopped. Press Enter to exit.")

if __name__ == "__main__":
    main()
