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

map_file = 'default_map.db'

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

def get_user_info():
    group_name = input("Please enter your group's name: ")
    operator_name = input("Please enter the operator's name: ")
    task = input("Please enter the name of your task (optional-leave blank): ")
    return group_name, operator_name, task

def main():
    global map_file

    signal.signal(signal.SIGINT, signal_handler)

    # Set the current time for the filename
    now = datetime.datetime.now().strftime("%F__%H_%M_%S")

    # Set ROS parameters
    subprocess.run(['rosparam', 'set', 'use_sim_time', 'false'])

    newMap = input("Would you like to create a new map? [Y/n]").lower() == 'y'

	# Directory of this script
    script_dir = os.path.dirname(os.path.realpath(__file__))

    if newMap:
        map_file = input("Enter the name of the map file to create (without extension): ")
        map_file_path = os.path.join(script_dir, 'saved_maps', f'{map_file}.db')

        print("Creating new map...")
        subprocess.Popen(['roslaunch', './launch/realsense_create_new_map.launch', 'database_path:=' + map_file_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        # Show saved maps
        print("These are the saved maps:")
        saved_maps = os.listdir(os.path.join(script_dir, 'saved_maps'))
        print(saved_maps)

        map_file = input("Enter the name of the map file to load (without extension): ")
        map_file_path = os.path.join(script_dir, 'saved_maps', f'{map_file}.db')

        print("Loading map from file...")
        subprocess.Popen(['roslaunch', './launch/realsense_load_from_map.launch', 'database_path:=' + map_file_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


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

    delete = input("Keep this recording? [Y/n]").lower() == 'n'
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
        "task": task if len(task) > 0 else "empty",
        "map_file": map_file
    }

    data.append(entry)

    with open (file_name, 'w') as file:
        json.dump(data, file, indent=4)

    input("Recording stopped. Press Enter to exit.")

if __name__ == "__main__":
    main()
