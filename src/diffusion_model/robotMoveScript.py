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

import transforms3d as tf3d

import h5py
from numpy import linalg as LA



class SawyerEnv():
    def __init__(self) -> None:
        rospy.init_node('go_to_cartesian_pose_py')
        self.limb = Limb()
        self.tip_name = "right_hand"

        #self.pipeline = rs.pipeline()
        #self.config = rs.config()
       
        # change coordinates if necessary
        # self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        # self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start the camera
        #self.pipeline.start(self.config)
        self.rate = rospy.Rate(10)
   
    # closes the camera
    def reset(self):
        self.pipeline.stop()

    # referred to this: # https://github.com/RethinkRobotics/intera_sdk/blob/master/intera_examples/scripts/ik_service_client.py
    def go_to_cartesian(self, x1, y1, z1, q1, q2, q3, q4, tip_name="right_hand"):    
        pose = Pose()
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
   
        self.limb.set_joint_positions(joint_angles)
       

    def step(self,action):
        #make translation and quaternion into a matrix
        prev = self.limb.endpoint_pose()
        prev_pose = np.eye(4)
        rotmat0 = tf3d.quaternions.quat2mat(prev['orientation'])
        prev_pose[:3,:3] = rotmat0
        prev_pose[0,3] = prev['position'].x
        prev_pose[1,3] = prev['position'].y
        prev_pose[2,3] = prev['position'].z
    
        rotmat1 = tf3d.quaternions.quat2mat([action[1][3],action[1][0],action[1][1],action[1][2]])
        homomat = np.eye(4)
        homomat[:3,:3] = rotmat1
        homomat[0,3] = action[0][0]
        homomat[1,3] = action[0][1]
        homomat[2,3] = action[0][2]
    
        if prev_pose is not None:
            new_pose = np.matmul(homomat,prev_pose)
    
        rotmat2 = new_pose[:3, :3]
        quaternion = tf3d.quaternions.mat2quat(rotmat2)
    
        self.go_to_cartesian(new_pose[0, 3], new_pose[1, 3], new_pose[2, 3], quaternion[0], quaternion[1], quaternion[2], quaternion[3])
    
        self.rate.sleep()

        image = self.capture_image()
        return {"new_pose": new_pose, "new_image": image}
   
    def replay_bag(self,file1):
        rate = rospy.Rate(10)

        f = h5py.File(file1, 'r')
    
        for g in f.keys():
            group = f[g]
            for i in range(len(group['actions'])):
                self.step(group['actions'][i])
   
    # referred to this: https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/opencv_viewer_example.py
    def receieve_image(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))
       
        # If needed, return both images
        return images
     
    '''
    def step(self, action):
        # TODO : here apply the action, use go_to_cartesian
       
        # TODO: return the observation
        new_coords = self.go_to_cartesian(action[0], action[1], action[2], action[3], action[4], action[5], action[6])
        #image = self.capture_image()
        self.rate.sleep()
        # return {"new_pose": new_coords, "new_image": image}'''
       
def run_episode(policy, env):
    pass
   
def normalize(quaternion):
    quat = np.array(quaternion)
    quat = LA.norm(quat)
   
    return quat

if __name__ == '__main__':
    env = SawyerEnv()
    rate = rospy.Rate(10)

    env.replay_bag("dummy_data1.hdf5")
   
    rate.sleep()