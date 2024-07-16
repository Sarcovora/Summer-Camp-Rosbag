import os
import subprocess
import sys
import json
import map_recreate
import demo_offline_SLAM
from os.path import exists


script_dir = os.path.dirname(os.path.realpath(__file__))
dict_path = os.path.join(script_dir, 'data', 'bag_dict.json')
maps_dir = os.path.join(script_dir, 'maps')
recreated_maps_dir = os.path.join(maps_dir, 'recreated_maps')

def main():
    with open(dict_path, 'r') as file:
        bag_list = json.load(file)

        

        for bag_info in bag_list:
            hdf5_path = os.path.join(script_dir, 'data', 'hdf5_files', f"{bag_info.get('bag_name')}")
            bag_path = os.path.join(script_dir, 'data', 'saved_demo_bags', f"{bag_info.get('bag_name')}")
            map_name = bag_info.get('map_file')
            map_file_path = os.path.join(recreated_maps_dir, f'{map_name}.db')
            if exists(bag_path):
                if (not exists(map_file_path)):
                    print("The corresponding map file has not been recreated. Exiting.")
                    sys.exit(1)
                print(f"Processing bag file: {bag_path}")
                # map_recreate.recreate_mapping(map_file_path=map_file_path, bag_path=bag_path, bag_playback_rate=0.5)
                
                demo_offline_SLAM.rebag(source_map_file_path=map_file_path, bag_path=bag_path, bag_playback_rate=0.25)
            else: 
                print("Rosbag file not found. Exiting. ")
                sys.exit(1)
            sys.exit(1)


if __name__ == "__main__":
    main()
        


    



