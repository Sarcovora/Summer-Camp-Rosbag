#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import tf2_ros
import tf2_geometry_msgs
import geometry_msgs.msg

from collections import deque

from numpy.linalg import inv

from tf.transformations import quaternion_matrix
from tf.transformations import euler_from_matrix

# /camera/aligned_depth_to_color/image_raw
# /camera/color/image_raw

# CORRECT
# /camera/color/image_raw/compressed
# /camera/aligned_depth_to_color/image_raw/compressed

# /camera/aligned_depth_to_color/camera_info
# /camera/color/camera_info
# /clock
# /rosout
# /rosout_agg

# /tf
# /tf_static

colorDeque = deque()
depthDeque = deque()

def color_image_callback(data):
    try:
        bridge = CvBridge()
        # Convert the ROS Image message to OpenCV2
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
        colorDeque.append(cv_image)
        # Display image
        # cv2.imshow("Color Image", cv_image)
        # cv2.waitKey(1)
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

def depth_image_callback(data):
    try:
        bridge = CvBridge()
        # Convert the ROS Image message to OpenCV2
        cv_image = bridge.imgmsg_to_cv2(data, "16UC1")
        depthDeque.append(cv_image)
        # Display image
        # cv2.imshow("Depth Image", cv_image)
        # cv2.waitKey(1)
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

def main():
    rospy.init_node('image_subscriber_node', anonymous=True)

    rospy.Subscriber('/camera/color/image_raw', Image, color_image_callback)
    rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, depth_image_callback)

    # tf lookup setup
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    data = []

    rate = rospy.Rate(30)
    prevtrans = tf_buffer.lookup_transform('map', 'map', rospy.Time(0))
    while not rospy.is_shutdown():
        # tf lookup here
        try:
            trans = tf_buffer.lookup_transform('camera_link', 'map', rospy.Time(0))
            translation = trans.transform.translation
            rotation = trans.transform.rotation

            prevtranslation = prevtrans.transform.translation
            prevrotation = prevtrans.transform.rotation
            # print("Translation: ", type(translation))
            # print("Rotation: ", rotation)
            if (colorDeque and depthDeque):
                currRot = quaternion_matrix([rotation.x, rotation.y, rotation.z, rotation.w])
                prevRot = quaternion_matrix([prevrotation.x, prevrotation.y, prevrotation.z, prevrotation.w])

                currRot[:3, -1] = [translation.x, translation.y, translation.z]
                prevRot[:3, -1] = [prevtranslation.x, prevtranslation.y, prevtranslation.z]

                deltaHomo = inv(prevRot) @ currRot

                # print(deltaHomo)

                # get the Euler and Translation values from homgenous matrix
                translation = deltaHomo[:3, -1]
                rotationMat = deltaHomo[:3, :3]

                euler = euler_from_matrix(rotationMat, 'rxyz')

                # print(translation)
                # print(euler)

                data.append({'color': colorDeque[-1], 'depth': depthDeque[-1], 'tf_relative_homo': currRot, 'tf_delta_homo': deltaHomo})
                print(data[-1])
            else:
                print("No if was entered ;(")
            prevtrans = trans

            print("translation and rotation successfully parsed")
        except Exception as e:
            print("oops:", e)

        rate.sleep()

    rospy.spin()
    print("finished reading bag!")

    # Close all OpenCV windows
    print(data)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
