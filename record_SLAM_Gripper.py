#!/usr/bin/env python3

import os
import signal
import subprocess
import time
import datetime
import threading
import json

recording = True

file_name = 'bag_dict.json'

def signal_handler(sig, frame):
    global recording
    print("\nStopping recording...")
    recording = False

def record_rosbag(now):
    global recording
    process = subprocess.Popen([
        'rosbag', 'record', '-O', f'SLAM_{now}.bag', '-b', '0',
        '/camera/aligned_depth_to_color/camera_info',
        '/camera/aligned_depth_to_color/image_raw/compressed',
        '/camera/color/camera_info',
        '/camera/color/image_raw/compressed',
        '/camera/imu',
        '/camera/imu_info',
        '/tf_static',
        '/tf',
        '/bariflex'
    ])
    while recording:
        time.sleep(1)
    process.terminate()
    process.wait()

def get_user_info():
    group_name = input("Please enter your group's name: ")
    operator_name = input("Please enter the operator's name: ")
    task = input("Please enter the name of your task (optional-leave blank): ")
    return group_name, operator_name, task

def main():
    signal.signal(signal.SIGINT, signal_handler)

    # Set the current time for the filename
    now = datetime.datetime.now().strftime("%F__%H_%M_%S")

    # Set ROS parameters
    subprocess.run(['rosparam', 'set', 'use_sim_time', 'false'])

    # Launch the RealSense camera node
    subprocess.Popen(['roslaunch', 'realsense2_camera', 'opensource_tracking.launch'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # subprocess.Popen(['roslaunch', 'realsense2_camera', 'opensource_tracking_from_map.launch'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # subprocess.Popen(['roslaunch', 'realsense2_camera', 'opensource_tracking_SLAMless.launch'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


    input("Press Enter to start recording\n")
    print("Press 's' and Enter to stop recording")

    # Start recording in a separate thread
    record_thread = threading.Thread(target=record_rosbag, args=(now,))
    record_thread.start()

    while recording:
        if input() == 's':
            signal_handler(None, None)

    record_thread.join()

    # Kill the rest of the processes
    print("Killing rosnode processes")
    subprocess.run(['rosnode', 'kill', '--all'])

    bag_name = 'SLAM_' + now + '.bag'

    delete = input("Keep this recording? [Y/n]") == 'n'
    if (delete):
        print("Deleting bag...")
        os.remove(bag_name)
        print(bag_name, 'deleted.')
        return

    if os.path.exists(file_name):
        with open(file_name, 'r') as file:
            data = json.load(file)
    else:
        data = []

    group, operator, task = get_user_info()

    entry = {
        "bag_name": bag_name,
        "group_name": group,
        "operator_name": operator,
        "time_stamp": now,
        "task": task if len(task) > 0 else "empty"
    }

    data.append(entry)

    with open (file_name, 'w') as file:
        json.dump(data, file, indent=4)

    input("Recording stopped. Press Enter to exit.")

if __name__ == "__main__":
    main()
