#!/usr/bin/env python3
import os
import signal
import subprocess
import time
import sys
import rospy
from os.path import exists
import yaml
import time

import re
import rosbag
import message_filters
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros
import tf2_geometry_msgs
import geometry_msgs.msg
import argparse
from tf import ExtrapolationException

from numpy.linalg import inv

import h5py

from tf.transformations import quaternion_matrix
from tf.transformations import quaternion_from_matrix

recording = True
offline = True
terminated = False

bag_playback_rate = 0.25

script_dir = os.path.dirname(os.path.realpath(__file__))
hdf5_name = "rosbag.hdf5"
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

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

script_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(script_dir, 'data')

uber_color_arr, uber_depth_arr, uber_pos_arr, uber_action_arr = [], [], [], []

tf_buffer = None
tf_listener = None

prev_pose = None

def sync_callback(color, depth, bariflex):
    print("Status: starting callback")
    global uber_color_arr, uber_depth_arr, uber_pos_arr, uber_action_arr
    global tf_buffer, tf_listener
    global prev_pose
    global terminated

    try:
        bridge = CvBridge()
        cv_image = bridge.compressed_imgmsg_to_cv2(color, "bgr8")
        cv_image = cv2.resize(cv_image, (160, 90), interpolation=cv2.INTER_AREA)
        rgb_arr = np.asarray(cv_image)        

    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))
        
    try:
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(depth, "16UC1")
        cv_image = cv2.resize(cv_image, (160, 90), interpolation=cv2.INTER_AREA)
        depth_arr = np.asarray(cv_image)


        # cv2.imshow("Depth Image", cv_image)
        # cv2.waitKey(1)
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

    regex = [float(x.group()) for x in re.finditer(r"-{0,1}\d+\.\d+", bariflex.data)]

    try:
        # tf lookup
        # tf_listener.waitForTransform("/camera_link", "/odom", now, rospy.Duration(2))
        trans = tf_buffer.lookup_transform('camera_link', 'odom', rospy.Time())
        translation = trans.transform.translation
        rotation = trans.transform.rotation

        # compute the 4x4 transform matrix representing the pose in the map
        mat = quaternion_matrix([rotation.x, rotation.y, rotation.z, rotation.w])
        transform_matrix = np.identity(4)
        transform_matrix[:3, :3] = mat[:3, :3]
        transform_matrix[0, 3] = translation.x
        transform_matrix[1, 3] = translation.y
        transform_matrix[2, 3] = translation.z
        # compute the relative pose between this pose and the previous pose, prev_pose
        if prev_pose is not None:
            inv_prev = inv(prev_pose)
            rel_pose = np.matmul(inv_prev, transform_matrix)
        else:
            rel_pose = np.eye(4)
        prev_pose = transform_matrix
        deltaTrans = [0, 0, 0]
        deltaTrans[0] = rel_pose[0, 3]
        deltaTrans[1] = rel_pose[1, 3]
        deltaTrans[2] = rel_pose[2, 3]
        deltaQuat = [0, 0, 0, 1]
        deltaQuat = quaternion_from_matrix(rel_pose)

        if np.linalg.norm(deltaTrans) >= 10:
            terminated = True
            return 

        # record the relative pose together with the most recent depth and color image received by subscribers
        if rgb_arr is not None and depth_arr is not None and regex is not None:
                uber_color_arr.append(rgb_arr)
                uber_depth_arr.append(depth_arr)
                uber_pos_arr.append(regex[1])
                uber_action_arr.append([*deltaTrans, *deltaQuat, regex[0]])
                print("Status: appended data")
    except Exception as e:
        print("oops: ", e)

def listener_sync(duration):
    global tf_buffer, tf_listener
    global terminated
    
    start_time = time.time()
    end_time = start_time + duration
    rospy.init_node("hdf5_parser", anonymous=True)
    
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    color_sub = message_filters.Subscriber("/camera/color/image_raw/compressed", CompressedImage)
    depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)
    bariflex_sub = message_filters.Subscriber("/bariflex", String)

    sync = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub, bariflex_sub], queue_size=10000, slop=1, allow_headerless=True)
    sync.registerCallback(sync_callback)
    # rate = rospy.Rate(10)
    while time.time() <= end_time:
        # print(f"Status: sleeping - {time.time()}, {end_time}")
        # rate.sleep()
        if terminated:
            print("script terminated: transform out of range")
            break
        time.sleep(0.1)

def write_hdf5():
    global uber_color_arr, uber_depth_arr, uber_pos_arr, uber_action_arr
    global hdf5_name

    if len(uber_action_arr) == 0:
        print('no transitions found')
        return 
    
    if np.isnan(uber_action_arr).any():
        print('caught nans')
        return 
    
    if (np.linalg.norm(np.array(uber_action_arr)[:,:3], axis=1) > 10).any():
        print('caught inf')
        return 

    with h5py.File(os.path.join(data_dir, hdf5_name), "a") as hdf5_file:
        
        if 'data' not in hdf5_file:
            data_group = hdf5_file.create_group('data')
        else:
            data_group = hdf5_file['data']

        num_demos = len(data_group.keys())
        name = f"demo_{num_demos}"
        print(f"Writing hdf5 group: {name}")
        group = data_group.create_group(name)
        # hdf5_file
        group.attrs['num_samples'] = len(uber_action_arr)
        group.create_dataset("obs/color", data=np.array(uber_color_arr))
        group.create_dataset("obs/depth", data=np.array(uber_depth_arr).reshape(len(uber_depth_arr), 160, 90, 1) // 256 )
        group.create_dataset("obs/pos", data=np.array(uber_pos_arr).reshape(len(uber_pos_arr), 1))
        group.create_dataset("actions", data=np.array(uber_action_arr))
    
    print(f'writing {len(uber_action_arr)} transitions')

def signal_handler(sig, frame):
    sys.exit(1)
    # global recording
    # print("\nStopping recording...")
    # recording = False

def rebag(source_map_file_path=None, bag_path=None, bag_playback_rate=0.25):
    # global source_map_file_path, bag_path, bag_playback_rate

    signal.signal(signal.SIGINT, signal_handler)

    info_dict = yaml.load(subprocess.Popen(['rosbag', 'info', '--yaml', bag_path], stdout=subprocess.PIPE).communicate()[0], Loader=yaml.FullLoader)
    bag_duration = info_dict['duration'] / bag_playback_rate

    subprocess.run(['rosparam', 'set', 'use_sim_time', 'true'])

    subprocess.Popen(['roslaunch', './launch/realsense_load_from_map.launch', 'database_path:=' + source_map_file_path, 'offline:=true'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    print("Running image_transport")

    # Run the image_transport republish commands
    subprocess.Popen(['rosrun', 'image_transport', 'republish', 'compressed', 'in:=/camera/color/image_raw', 'raw', 'out:=/camera/color/image_raw'])
    
    if not bag_path:
        print("ERR: Couldn't find rosbag file. Exiting.")
        sys.exit(1)

    listener_sync(bag_duration + 1)

    print("Killing rosnode processes")
    subprocess.run(['rosnode', 'kill', '--all'])

    print("writing start")
    write_hdf5()
    print("things were written")

def recreate_mapping(map_file_path=None, bag_path=None, bag_playback_rate=0.25):

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
    global map_name, script_dir, bag_path, offline, source_map_file_path
    global hdf5_name
    
    bag_path = sys.argv[1] if len(sys.argv) > 1 else None
    hdf5_name = sys.argv[2] if len(sys.argv) > 2 else "rosbag.hdf5"
    bag_rate = 0.25

    if bag_path:
        map_name = os.path.basename(bag_path).split('_')[1]
        source_map_file_path = os.path.join(recreated_maps_dir, f'{map_name}.db')
        print(source_map_file_path)
        if not exists(source_map_file_path):
            mapping_bag_path = None
            print("The corresponding map file has not been recreated. Exiting.")
            # for file in os.listdir(saved_mapping_dir):
            #     if f"{map_name}" in file:
            #         mapping_bag_path = file   
            # if mapping_bag_path is not None:
            #     recreate_mapping(map_file_path=source_map_file_path, bag_path=mapping_bag_path, bag_playback_rate=bag_rate)
            # else:
            #     print("missing mapping bag path")
            sys.exit(1)
        else:
            rebag(source_map_file_path=source_map_file_path, bag_path=bag_path, bag_playback_rate=bag_rate)
    else:
        print("No rosbag file specified. Exiting.")
        sys.exit(1)

if __name__ == "__main__":
    main()