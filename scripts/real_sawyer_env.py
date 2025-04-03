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

from std_msgs.msg import UInt16


class SawyerEnv():

    def __init__(self) -> None:
        rospy.init_node('go_to_cartesian_pose_py')
        self.limb = Limb()
        self.tip_name = "right_hand"

        self.pub = rospy.Publisher('/bariflex_motion', UInt16, queue_size=10)
        rospy.Subscriber('/bariflex', String, self.callback_fn)

        self.pipeline = rs.pipeline()
        self.config = rs.config()
       
        # change coordinates if necessary
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start the camera
        self.pipeline.start(self.config)
        self.rate = rospy.Rate(10)
        self.horizon = 500
        self.step_counter = 0 
        self.bariflex_state = 0.

    def callback_fn(self, msg):
        # current = float(re.match(r".Iq-*\d+\.\d+)", msg.data).group(1))
        # desire = float(re.match(r".des-*\d+\.\d+)", msg.data).group(1))
        # position = float(re.match(r".pos-*\d+\.\d+)", msg.data).group(1))

        if msg.data[-5] == '-':
            current = (float(msg.data[-5:]))
        else:
            current = (float(msg.data[-4:]))

        if msg.data[4] == '-':
            desire = (float(msg.data[4:9]))
        else:
            desire = (float(msg.data[4:8]))

        if msg.data[14] == '-':
            position = (float(msg.data[14:19]))
        else:
            position = (float(msg.data[14:18]))
        
        # print("des: " + desire + " pos: " + position + " Iq: " + current)
        self.bariflex_state = position

    def get_bariflex_state(self):
        return self.bariflex_state
    
    def save_pose(self):
        global neturalx, neturaly, neturalz, netural2x, netural2y, netural2z, netural2w
        tempVar = env.limb.endpoint_pose()["position"]
        tempVar2 = env.limb.endpoint_pose()["orientation"]
        neturalx, neturaly, neturalz, netural2x, netural2y, netural2z, netural2w = tempVar.x, tempVar.y, tempVar.z, tempVar2.x, tempVar2.y, tempVar2.z, tempVar2.w

    # closes the camera
    def reset(self):
        self.step_counter = 0 

        prev = self.limb.endpoint_pose()
        prev_pose = np.eye(4)
        quat = prev['orientation']
        quat = quat[3],quat[0],quat[1],quat[2]
        rotmat0 = tf3d.quaternions.quat2mat(quat)
        prev_pose[:3,:3] = rotmat0
        prev_pose[0,3] = prev['position'].x
        prev_pose[1,3] = prev['position'].y
        prev_pose[2,3] = prev['position'].z
        self.setpoint = prev_pose

        # self.go_to_cartesian(neturalx, neturaly, neturalz, netural2x, netural2y, netural2z, netural2w)

        # self.limb.move_to_neutral(self, timeout=15.0, speed=0.3)
        # self.pipeline.stop()
        return self.get_obs()

    # referred to this: # https://github.com/RethinkRobotics/intera_sdk/blob/master/intera_examples/scripts/ik_service_client.py
    def go_to_cartesian(self, x1, y1, z1, q1, q2, q3, q4, tip_name="right_hand"):    
        pose = Pose()        # return dictionary of "color", "depth"
        pose.position.x = x1
        pose.position.y = y1
        pose.position.z = z1
        pose.orientation.x = q1
        pose.orientation.y = q2
        pose.orientation.z = q3
        pose.orientation.w = q4
        print("endpose",self.limb.endpoint_pose())
        print("params", pose, tip_name)
        joint_angles = self.limb.ik_request(pose, tip_name)

        if joint_angles:
            self.limb.set_joint_positions(joint_angles)
        else:
            print('bad ik')
   
       
    def handle_gripper_action(self, gripper_act):
        # TODO: for group 6
        # TODO: publish to bariflex_motion topic 
        # TODO: make publisher above in __init__ 
        # TODO: convert action from policy to integer: 
        int_val = int(gripper_act)
        if int_val<0.5:
            self.pub.publish(2)
        else:
            self.pub.publsh(1)
        # 1: open bariflex (buttons will be disabled)
        # 2: close bariflex (buttons will be disabled)
        
        # TODO: init_node once during __init__ 
        # TODO: make the publisher once during __init__ 
        # TODO: publish one message rather than while loop
        # thanks :) 
        pass

    def get_obs(self):
        obs = {}
        obs = self.receive_image()
        obs["pos"] = np.array(self.bariflex_state)
        return obs

    def step(self,action):

        # ros: xyzw
        # tf3d: wxyz
        # action: xyzw, from ros 
     
        self.step_counter += 1

        #make translation and quaternion into a matrix
        xyz = action[:3]  
        xyz = np.clip(xyz, -0.05, 0.05)

        quat = action[3:7]
        mag = np.linalg.norm(quat)
        quat = quat / mag 

        rotmat1 = tf3d.quaternions.quat2mat([quat[3],quat[0],quat[1],quat[2]])
        homomat = np.eye(4)
        homomat[:3,:3] = rotmat1
        homomat[0,3] = xyz[0]
        homomat[1,3] = xyz[1]
        homomat[2,3] = xyz[2]
    
        new_pose = np.matmul(self.setpoint, homomat)
    
        rotmat2 = new_pose[:3, :3]
        quaternion = tf3d.quaternions.mat2quat(rotmat2)
        mag = np.linalg.norm(quaternion)
        quaternion = quaternion / mag 
        
        self.go_to_cartesian(new_pose[0, 3], new_pose[1, 3], new_pose[2, 3], quaternion[1], quaternion[2], quaternion[3], quaternion[0])
        self.handle_gripper_action(action[-1])
        self.setpoint = new_pose

        self.rate.sleep()

        reward = 0
        done = (self.step_counter > self.horizon)   
        info = {}
        obs = self.get_obs()
        return obs, reward, done, info 
        
           # return dictionary of "color", "depth"
    def replay_bag(self,file1):
        rate = rospy.Rate(10)

        f = h5py.File(file1, 'r')
    
        for g in f.keys(): 
            group = f[g]
            for i in range(len(group['actions'])):
                self.step(group['actions'][i])
   
    # referred to this: https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/opencv_viewer_example.py
    def receive_image(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        color_image = cv2.resize(color_image, (160, 90), interpolation=cv2.INTER_AREA)
        depth_image = cv2.resize(depth_image, (160, 90), interpolation=cv2.INTER_AREA)

        depth_arr = np.asarray(depth_image, dtype=float) / 65535
        color_arr = np.asarray(color_image, dtype=float) / 256

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=UInt16,
        image = {"color": color_arr.reshape(3, 90, 160),
                 "depth": depth_arr.reshape(1, 160, 90) }
        
        return image
 


def normalize(quaternion):
    quat = np.array(quaternion)
    quat = LA.norm(quat)
   
    return quat# TODO: 

if __name__ == '__main__':
    env = SawyerEnv()

    #saves current pose of robot    
    # env.save_pose()
    act = np.zeros(8)
    act[-2] = 1
    obs, rew, done, info = env.step(act)

    # rate = rospy.Rate(10)
    # tempVar = env.limb.endpoint_pose()["position"]
    # tempVar2 = env.limb.endpoint_pose()["orientation"]
    #moves robot slightly
    # for i in range(10):
        # env.go_to_cartesian(tempVar.x, tempVar.y, tempVar.z - .1, tempVar2.x, tempVar2.y, tempVar2.z, tempVar2.w)
    #resets to saved pose of robot
    # env.reset()
    
    # rate.sleep()

    # env.replay_bag("dummy_data1.hdf5")
