#!/usr/bin/env python3

import os
import signal
import subprocess
import time
import datetime
import threading
import json
import sys
from os.path import exists

recording = True
offline = True

bag_playback_rate = 0.5

script_dir = os.path.dirname(os.path.realpath(__file__))

data_dir = os.path.join(script_dir, 'data')
maps_dir = os.path.join(script_dir, 'maps')

saved_maps_dir = os.path.join(maps_dir, 'saved_maps')
recreated_maps_dir = os.path.join(maps_dir, 'recreated_maps')

saved_demos_dir = os.path.join(data_dir, 'saved_demo_bags')
saved_mapping_dir = os.path.join(data_dir, 'saved_mapping_bags')
recreated_bags_dir = os.path.join(data_dir, 'recreated_demo_bags')

dict_path = os.path.join(script_dir, 'data', 'bag_dict.json')

os.makedirs(data_dir, exist_ok=True)
os.makedirs(maps_dir, exist_ok=True)

os.makedirs(saved_maps_dir, exist_ok=True)
os.makedirs(recreated_maps_dir, exist_ok=True)

os.makedirs(saved_demos_dir, exist_ok=True)
os.makedirs(saved_mapping_dir, exist_ok=True)
os.makedirs(recreated_bags_dir, exist_ok=True)

source_map_file_path = 'default_map_file'
bag_path = 'default_bag_path'

def generate_bag_path():
    global data_dir, dict_path, bag_name, map_name
    return os.path.join(recreated_bags_dir, f'recreatedDemo_{os.path.basename(bag_path)}')

def signal_handler(sig, frame):
    global recording
    print("\nStopping recording...")
    recording = False

def record_rosbag():
    global recording, bag_path
    target_bag_path = generate_bag_path()
    print("target_bag_path:", target_bag_path)
    process = subprocess.Popen([
        'rosbag', 'record', '-O', target_bag_path, '-b', '0',
        '/camera/aligned_depth_to_color/camera_info',
        '/camera/aligned_depth_to_color/image_raw',
        '/camera/aligned_depth_to_color/image_raw/compressedDepth',
        '/camera/color/camera_info',
        '/camera/color/image_raw/compressed',
        '/camera/imu',
        '/camera/gyro/imu_info',
        '/camera/accel/imu_info',
        '/tf_static',
        '/tf',
        '/bariflex',
        '/bariflex_motion'
    ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print("started recording")
    while recording:
        time.sleep(1)
    process.terminate()
    process.wait()

def rebag():
    global source_map_file_path, bag_path, bag_playback_rate

    signal.signal(signal.SIGINT, signal_handler)

    subprocess.run(['rosparam', 'set', 'use_sim_time', 'true'])

    subprocess.Popen(['roslaunch', './launch/realsense_load_from_map.launch', 'database_path:=' + source_map_file_path, 'offline:=true'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    print("Running image_transport")

    # Run the image_transport republish commands
    subprocess.Popen(['rosrun', 'image_transport', 'republish', 'compressed', 'in:=/camera/color/image_raw', 'raw', 'out:=/camera/color/image_raw'])

    record_thread = threading.Thread(target=record_rosbag)
    record_thread.start()

    if bag_path:
        subprocess.run(['rosbag', 'play', bag_path, '--rate', str(bag_playback_rate), '--clock'])
    else:
        print("ERR: Couldn't find rosbag file. Exiting.")
        sys.exit(1)

    record_thread.join()
    input("Press Enter to exit...")
    print("Killing rosnode processes")
    subprocess.run(['rosnode', 'kill', '--all'])

def main():
    global map_name, script_dir, bag_name, bag_path, offline, source_map_file_path

    bag_path = sys.argv[1] if len(sys.argv) > 1 else ""
    if bag_path:
        map_name = os.path.basename(bag_path).split('_')[1]
        print('map_name:', map_name)
        source_map_file_path = os.path.join(recreated_maps_dir, f'{map_name}.db')
        if (not exists(source_map_file_path)):
            print("The corresponding map file has not been recreated. Exiting.")
            sys.exit(1)
        rebag()
    else:
        print("No rosbag file specified. Exiting.")
        sys.exit(1)

if __name__ == "__main__":
    main()
