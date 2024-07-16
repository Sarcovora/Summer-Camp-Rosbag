import rospy
import argparse
import intera_interface
from intera_motion_interface import (
    MotionTrajectory,
    MotionWaypoint,
    MotionWaypointOptions
)
# from intera_motion_msgs.msg import TrajectoryOptions
from geometry_msgs.msg import PoseStamped, Pose
from intera_interface import Limb
import numpy as np
import math
import time
import pyrealsense2 as rs
import cv2
import re
from std_msgs.msg import String
from collections import deque
import transforms3d as tf3d
import h5py
from numpy import linalg as LA
from robotMoveScript import SawyerEnv
import json


dummy_class = SawyerEnv()

dataset = []
dataset.append(dummy_class.save_pose())

with open("home_pose.json", "w") as final:
  json.dumps(dataset, final)
