#!/usr/bin/env python3

import os
import signal
import subprocess
import time
import datetime
import threading
import json
import sys

recording = True
offline = False

bag_playback_rate = 0.5

script_dir = os.path.dirname(os.path.realpath(__file__))

data_dir = os.path.join(script_dir, 'data')
saved_maps_dir = os.path.join(script_dir, 'saved_maps')
saved_demos_dir = os.path.join(data_dir, 'saved_demos')
saved_mapping_dir = os.path.join(data_dir, 'saved_mapping')

dict_path = os.path.join(script_dir, 'data', 'bag_dict.json')

os.makedirs(data_dir, exist_ok=True)
os.makedirs(saved_maps_dir, exist_ok=True)
os.makedirs(saved_demos_dir, exist_ok=True)
os.makedirs(saved_mapping_dir, exist_ok=True)

bag_name = 'default_bag'
bag_path = 'default_path'
map_name = 'default_map'
map_file_path = 'default_map_file'
mappingStage = True

def generate_bag_path(now):
    global data_dir, dict_path, bag_name, map_name, mappingStage
    if (mappingStage):
        bag_name = f'mapping_{map_name}_{now}.bag'
        return os.path.join(saved_mapping_dir, f'mapping_{map_name}_{now}.bag')
    else:
        bag_name = f'demo_{map_name}_{now}.bag'
        return os.path.join(saved_demos_dir, f'demo_{map_name}_{now}.bag')

def signal_handler(sig, frame):
    global recording
    print("\nStopping recording...")
    recording = False

def record_rosbag(now):
    global recording, bag_path
    bag_path = generate_bag_path(now)
    process = subprocess.Popen([
        'rosbag', 'record', '-O', bag_path, '-b', '0',
        '/camera/aligned_depth_to_color/camera_info',
        '/camera/aligned_depth_to_color/image_raw/compressedDepth',
        '/camera/color/camera_info',
        '/camera/color/image_raw/compressed',
        '/camera/imu',
        '/camera/imu_info',
        '/tf_static',
        '/bariflex'
    ])
    while recording:
        time.sleep(1)
    process.terminate()
    process.wait()

def replay_offline_demo():
    global map_file_path, bag_path, bag_playback_rate

    signal.signal(signal.SIGINT, signal_handler)

    subprocess.run(['rosparam', 'set', 'use_sim_time', 'true'])

    subprocess.Popen(['roslaunch', './launch/realsense_load_from_map.launch', 'database_path:=' + map_file_path, 'offline:=true'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    print("Running image_transport")

    # Run the image_transport republish commands
    subprocess.Popen(['rosrun', 'image_transport', 'republish', 'compressed', 'in:=/camera/color/image_raw', 'raw', 'out:=/camera/color/image_raw'])
    subprocess.Popen(['rosrun', 'image_transport', 'republish', 'compressedDepth', 'in:=/camera/aligned_depth_to_color/image_raw', 'raw', 'out:=/camera/aligned_depth_to_color/image_raw'])

    input("Press Enter to start replaying\n")

    if bag_path:
        subprocess.run(['rosbag', 'play', bag_path, '--rate', str(bag_playback_rate), '--clock'])
    else:
        print("ERR: Couldn't find rosbag file. Exiting.")
        sys.exit(1)

    input("Press Enter to exit...")

    print("Killing rosnode processes")
    subprocess.run(['rosnode', 'kill', '--all'])

def get_user_info():
    group_name = input("Please enter your group's name: ")
    operator_name = input("Please enter the operator's name: ")
    task = input("Please enter the name of your task (optional-leave blank): ")
    return group_name, operator_name, task

def main():
    global map_name, script_dir, bag_name, bag_path, mappingStage, offline, map_file_path

    signal.signal(signal.SIGINT, signal_handler)

    # Set the current time for the filename
    now = datetime.datetime.now().strftime("%F__%H_%M_%S")

    # Set ROS parameters
    subprocess.run(['rosparam', 'set', 'use_sim_time', 'false'])

    mappingStage = input("Would you like to create a new map? [y/N] ").lower() == 'y'

    if mappingStage:
        map_name = input("Enter the name of the map file to create (without extension): ")
        # map_file_path = os.path.join(script_dir, 'saved_maps', f'{map_name}.db')
        map_file_path = os.path.join(saved_maps_dir, f'{map_name}.db')

        print("Creating new map...", map_file_path)
        subprocess.Popen(['roslaunch', './launch/realsense_create_new_map.launch', 'database_path:=' + map_file_path], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    else: # DEMO stage
        offline = input("Offline Slam? [y/N] ") == 'y'
        # Show saved maps
        print("These are the saved maps:")
        # saved_maps = os.listdir(os.path.join(script_dir, 'saved_maps'))
        saved_maps = os.listdir(saved_maps_dir)
        print(saved_maps)

        map_name = input("Enter the name of the map file to load (without extension): ")
        # map_file_path = os.path.join(script_dir, 'saved_maps', f'{map_name}.db')
        map_file_path = os.path.join(saved_maps_dir, f'{map_name}.db')

        print("Loading map from file...", map_file_path)

        if offline:
            print("Running offline")
            subprocess.Popen(['roslaunch', './launch/realsense_load_offline.launch', 'database_path:=' + map_file_path], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        else:
            subprocess.Popen(['roslaunch', './launch/realsense_load_from_map.launch', 'database_path:=' + map_file_path], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


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

    if (not mappingStage):
        replay_offline_demo()

        # after replay, ask to keep/delete
        delete = input("Keep this recording? [Y/n] ").lower() == 'n'
        if (delete):
            print("Deleting bag...")
            os.remove(bag_path)
            print(bag_path, 'deleted.')
            return

    if os.path.exists(dict_path):
        with open(dict_path, 'r') as file:
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
        "map_file": map_name
    }

    data.append(entry)

    with open (dict_path, 'w') as file:
        json.dump(data, file, indent=4)

    print("Recording saved to", bag_path)

    input("Recording stopped. Press Enter to exit.")

if __name__ == "__main__":
    main()
