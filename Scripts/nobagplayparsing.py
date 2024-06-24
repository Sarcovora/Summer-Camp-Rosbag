import rosbag
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
        cv2.imshow("Color Image", cv_image)
        cv2.waitKey(1)
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

def depth_image_callback(data):
    try:
        bridge = CvBridge()
        # Convert the ROS Image message to OpenCV2
        cv_image = bridge.imgmsg_to_cv2(data, "16UC1")
        depthDeque.append(cv_image)
        # Display image
        cv2.imshow("Depth Image", cv_image)
        cv2.waitKey(1)
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

# /camera/aligned_depth_to_color/image_raw
# /camera/color/image_raw
bag = rosbag.Bag('realsense.bag')

for topic, msg, t in bag.read_messages():
    # print(topic)
    # TODO make sure that topic is replaced by the slash
    # BUG possibly make sure sequential stuff works
    if topic == 'camera/aligned_depth_to_color/image_raw':
        print(topic, t)
        depth_image_callback(msg)
    elif topic == '/camera/color/image_raw':
        print(topic, t)
        color_image_callback(msg)
bag.close()