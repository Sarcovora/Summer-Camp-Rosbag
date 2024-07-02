#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, CompressedImage
import cv2
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros
import tf2_geometry_msgs
import geometry_msgs.msg
import rosbag
import argparse

from collections import deque

from numpy.linalg import inv

from tf.transformations import quaternion_matrix
from tf.transformations import euler_from_matrix

# Non Compressed Topics
# /camera/aligned_depth_to_color/image_raw
# /camera/color/image_raw

# Compression Topics
# /camera/color/image_raw/compressed
# /camera/aligned_depth_to_color/image_raw/compressed

# Topics Currently Not Needed
# /camera/aligned_depth_to_color/camera_info
# /camera/color/camera_info
# /clock
# /rosout
# /rosout_agg

# TF Topics
# /tf
# /tf_static

colorDeque = deque()
depthDeque = deque()

# denotes the number of elements added to the deque since the last TF update and data append
colorCyclesOffset = 0
depthCycleOffset = 0

def color_image_callback(data):
    global colorDeque, colorCyclesOffset
    try:
        bridge = CvBridge()
        # Convert the ROS CompressedImage message to OpenCV2
        cv_image = bridge.compressed_imgmsg_to_cv2(data, "bgr8")
        colorDeque.append(cv_image)
        colorCyclesOffset += 1

        # Display image
        # cv2.imshow("Color Image", cv_image)
        # cv2.waitKey(1)
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

def depth_image_callback(data):
    global depthDeque, depthCycleOffset
    try:
        bridge = CvBridge()
        # Convert the ROS CompressedImage message to OpenCV2
        cv_image = bridge.compressed_imgmsg_to_cv2(data, "16UC1")
        depthDeque.append(cv_image)
        depthCycleOffset += 1

        # Display image
        cv2.imshow("Depth Image", cv_image)
        cv2.waitKey(1)
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

def main():
    global colorDeque, colorCyclesOffset, depthDeque, depthCycleOffset

    rospy.init_node('image_subscriber_node', anonymous=True)

    parser = argparse.ArgumentParser(description='Process a ROS bag file.')
    parser.add_argument('bagfile', type=str, help='Path to the ROS bag file')
    args = parser.parse_args()

    bag_path = args.bagfile

    # tf lookup setup
    # tf_buffer = tf2_ros.Buffer()
    # tf_listener = tf2_ros.TransformListener(tf_buffer)

    data = []

    # rate = rospy.Rate(30)
    # prevTransform = tf_buffer.lookup_transform('map', 'map', rospy.Time(0))

    bag = rosbag.Bag(bag_path)

    for topic, msg, t in bag.read_messages():
        # Process the images
        if topic == '/camera/aligned_depth_to_color/image_raw/compressedDepth':
            depth_image_callback(msg)
        elif topic == '/camera/color/image_raw/compressed':
            color_image_callback(msg)

        # tf lookup stuff
				#     try:
				#         transform = tf_buffer.lookup_transform('camera_link', 'map', rospy.Time(0))
				#         translation = transform.transform.translation
				#         rotation = transform.transform.rotation
				#
				#         prevtranslation = prevTransform.transform.translation
				#         prevrotation = prevTransform.transform.rotation
				#
				#         if (colorDeque and depthDeque):
				#             currTransform = quaternion_matrix([rotation.x, rotation.y, rotation.z, rotation.w])
				#             prevTransform = quaternion_matrix([prevrotation.x, prevrotation.y, prevrotation.z, prevrotation.w])
				#
				#             currTransform[:3, -1] = [translation.x, translation.y, translation.z]
				#             prevTransform[:3, -1] = [prevtranslation.x, prevtranslation.y, prevtranslation.z]
				#
				#             deltaHomogenous = inv(prevTransform) @ currTransform
				#
				#             # get the Euler and Translation values from homgenous matrix
				#             translation = deltaHomogenous[:3, -1]
				#             rotationMat = deltaHomogenous[:3, :3]
				#
				#             euler = euler_from_matrix(rotationMat, 'rxyz')
				#
				# # print("translation", translation, "rotation", euler)
				#             print("color offset", colorCyclesOffset, "depth offset", depthCycleOffset)
				#
				#             data.append({'color': colorDeque[-1 * colorCyclesOffset], 'depth': depthDeque[-1 * depthCycleOffset], 'tf_relative_prev_homo': prevTransform, 'tf_relative_homo': currTransform, 'tf_delta_action_homo': deltaHomogenous})
				#             colorCyclesOffset = 0
				#             depthCycleOffset = 0
				#         else:
				#             print("No if was entered ;(")
				#         prevTransform = transform
				#
				#         print("translation and rotation successfully parsed")
				#     except Exception as e:
				#         print("oops:", e)
				#
				#     rate.sleep()

    bag.close()

    print("Finished reading bag!")

    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
