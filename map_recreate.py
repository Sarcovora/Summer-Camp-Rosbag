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

map_file_path = 'default_map_file'
bag_path = 'default_bag_path'

def signal_handler(sig, frame):
    global recording
    print("\nStopping recording...")
    recording = False

def recreate_mapping():
    global map_file_path, bag_path, bag_playback_rate

    signal.signal(signal.SIGINT, signal_handler)

    subprocess.run(['rosparam', 'set', 'use_sim_time', 'true'])

    # print('roslaunch', './launch/create_new_map.launch', 'database_path:=' + map_file_path, 'offline:=true')
    subprocess.Popen(['roslaunch', './launch/realsense_create_new_map.launch', 'database_path:=' + map_file_path, 'offline:=true'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    if bag_path:
        subprocess.run(['rosbag', 'play', bag_path, '--rate', str(bag_playback_rate), '--clock'])
    else:
        print("ERR: Couldn't find rosbag file. Exiting.")
        sys.exit(1)

    print("Killing rosnode processes")
    subprocess.run(['rosnode', 'kill', '--all'])

    print("Recreated map saved at:", map_file_path)

def main():
    global map_name, script_dir, bag_name, bag_path, offline, map_file_path

    bag_path = sys.argv[1] if len(sys.argv) > 1 else ""
    if bag_path:
        map_name = os.path.basename(bag_path).split('_')[1]
        print('map_name:', map_name)
        map_file_path = os.path.join(recreated_maps_dir, f'{map_name}.db')
        recreate_mapping()
    else:
        print("No rosbag file specified. Exiting.")
        sys.exit(1)

if __name__ == "__main__":
    main()
