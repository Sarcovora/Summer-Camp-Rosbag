#!/usr/bin/env python

import subprocess
import time
import signal
import sys
import os

script_dir = os.path.dirname(os.path.realpath(__file__))

data_dir = os.path.join(script_dir, 'data')
saved_maps_dir = os.path.join(script_dir, 'saved_maps')

os.makedirs(data_dir, exist_ok=True)
os.makedirs(saved_maps_dir, exist_ok=True)

# Function to handle SIGINT and kill all ros nodes
def signal_handler(sig, frame):
    subprocess.run(['rosnode', 'kill', '--all'])
    sys.exit(0)


def main():
	global script_dir, data_dir, saved_maps_dir

	signal.signal(signal.SIGINT, signal_handler)

	subprocess.run(['rosparam', 'set', 'use_sim_time', 'true'])

	# Show saved maps
	print("These are the saved maps:")
	saved_maps = os.listdir(saved_maps_dir)
	print(saved_maps)

	map_name = input("Enter the name of the map file to load (without extension): ")
	map_file_path = os.path.join(saved_maps_dir, f'{map_name}.db')

	subprocess.Popen(['roslaunch', './launch/realsense_load_from_map.launch', 'database_path:=' + map_file_path, 'offline:=true'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

	print("Running image_transport")

	# Run the image_transport republish commands
	subprocess.Popen(['rosrun', 'image_transport', 'republish', 'compressed', 'in:=/camera/color/image_raw', 'raw', 'out:=/camera/color/image_raw'])
	subprocess.Popen(['rosrun', 'image_transport', 'republish', 'compressedDepth', 'in:=/camera/aligned_depth_to_color/image_raw', 'raw', 'out:=/camera/aligned_depth_to_color/image_raw'])

	print("Press Ctrl^C to exit")

	input("Press Enter to start replaying\n")

	# Play the rosbag file
	rosbag_file = sys.argv[1] if len(sys.argv) > 1 else ""
	if rosbag_file:
		subprocess.run(['rosbag', 'play', rosbag_file, '--rate', '0.5', '--clock'])
	else:
		print("No rosbag file specified. Exiting.")
		sys.exit(1)

	input("Press Enter to exit...")

	print("Killing rosnode processes")
	subprocess.run(['rosnode', 'kill', '--all'])

if __name__ == "__main__":
    main()