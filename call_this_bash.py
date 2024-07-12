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
import threading

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

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

script_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(script_dir, 'data')

uber_color_arr, uber_depth_arr, uber_action_arr = [], [], []

tf_buffer = None
tf_listener = None

def sync_callback(color, depth, bariflex):
    global uber_color_arr, uber_depth_arr, uber_action_arr
    global tf_buffer, tf_listener

    try:
        bridge = CvBridge()
        cv_image = bridge.compressed_imgmsg_to_cv2(color, "bgr8")
        cv_image = cv2.resize(cv_image, (320, 180), interpolation=cv2.INTER_AREA)
        rgb_arr = np.asarray(cv_image)        

    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))
        
    try:
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(depth, "16UC1")
        cv_image = cv2.resize(cv_image, (320, 180), interpolation=cv2.INTER_AREA)
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
            rel_pose = transform_matrix
        prev_pose = transform_matrix
        deltaTrans = []
        deltaTrans[0] = rel_pose[0, 3]
        deltaTrans[1] = rel_pose[1, 3]
        deltaTrans[2] = rel_pose[2, 3]
        deltaQuat = []
        deltaQuat = quaternion_from_matrix(rel_pose)
        # record the relative pose together with the most recent depth and color image received by subscribers
        if rgb_arr is not None and depth_arr is not None and regex is not None:
                print("ifif")
                uber_color_arr.append(rgb_arr)
                uber_depth_arr.append(depth_arr)
                uber_action_arr.append([*deltaTrans, *deltaQuat, regex[0]])
                print("appends")
    except Exception as e:
        print("oops: ", e)

def listener_sync(duration):
    global tf_buffer, tf_listener
    
    start_time = time.time()
    end_time = start_time + duration
    rospy.init_node("hdf5_parser", anonymous=True)
    
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    color_sub = message_filters.Subscriber("/camera/color/image_raw/compressed", CompressedImage)
    depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)
    bariflex_sub = message_filters.Subscriber("/bariflex", String)

    sync = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub, bariflex_sub], 10, allow_headerless=True)
    sync.registerCallback(sync_callback)
    rate = rospy.Rate(10)
    while time.time() <= end_time:
        rate.sleep()

def write_hdf5(name):
    global uber_color_arr, uber_depth_arr, uber_action_arr
    with h5py.File(os.path.join(data_dir, "rosbag.hdf5"), "a") as hdf5_file:
        group = hdf5_file.create_group(name)
        group.create_dataset(f"{name}_color_images", data=np.array(uber_color_arr))
        group.create_dataset(f"{name}_depth_images", data=np.array(uber_depth_arr))
        group.create_dataset(f"{name}_actions", data=np.array(uber_action_arr))

def signal_handler(sig, frame):
    global recording
    print("\nStopping recording...")
    recording = False

def rebag(source_map_file_path=None, bag_path=None, bag_playback_rate=0.1):
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

    listener_sync(bag_duration + 0.5)

    print("Killing rosnode processes")
    subprocess.run(['rosnode', 'kill', '--all'])

    print("writing start")
    write_hdf5(os.path.basename(os.path.normpath(bag_path)))
    print("things were written")

def main():
    global map_name, script_dir, bag_path, offline, source_map_file_path

    bag_path = sys.argv[1] if len(sys.argv) > 1 else None
    bag_rate = 0.5

    if bag_path:
        map_name = os.path.basename(bag_path).split('_')[1]
        print('map_name:', map_name)
        source_map_file_path = os.path.join(recreated_maps_dir, f'{map_name}.db')
        if (not exists(source_map_file_path)):
            print("The corresponding map file has not been recreated. Exiting.")
            sys.exit(1)
        rebag(source_map_file_path=source_map_file_path, bag_path=bag_path, bag_playback_rate=0.5)
    else:
        print("No rosbag file specified. Exiting.")
        sys.exit(1)

if __name__ == "__main__":
    main()