# file imports
import camera
import dataset
import model
import train
from robotMoveScript import SawyerEnv

import cv2
import h5py
import intera_interface
import numpy as np
import pyrealsense2 as rs
import rospy
from matplotlib import pyplot as plt
#from pynput import keyboard

from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import math
import torch
import torch.nn as nn
import torchvision
import collections
import json
import zarr
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

from helper_functions import get_into_dataloader_format

class Runner():
    def __init__(self, side="right", obs_horizon=6, pred_horizon=4, #pred horizon must be a multiple of 4
                model_file="checkpoint.pth", stats_file="stats.json", device="cpu"):
        
        self.env = SawyerEnv()
        
        # rospy.init_node("Runner")
        self.limb = intera_interface.Limb(side)
        self.camera = camera.Camera()

        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.model_file = model_file
        self.stats_file = stats_file

        self.device = device

        with open(self.stats_file, "r") as file:
            stats_dict = json.load(file)
            self.pose_stats = stats_dict["pose_stats"]
            self.orientation_stats = stats_dict["orientation_stats"]
        
        self.image_stats = {"min" : 0, "max" : 255}

        self.nets, _ = model.load_pretrained(self.model_file, self.device, self.obs_horizon)
        self.nets.eval()

    def get_current_state(self):
        # endpoint_pose = self.limb.endpoint_pose()
        # pose, orientation = endpoint_pose["position"], endpoint_pose["orientation"]

        # pose = torch.tensor(list(pose)).reshape(1, -1)
        # orientation = torch.tensor(list(orientation)).reshape(1, -1)

        color_image = torch.from_numpy(self.camera.get_frame()).float()
        color_image = color_image.reshape(3, 640, 480)

        depth_image = torch.from_numpy(self.camera.get_depth_frame()).float()
        depth_image = depth_image.reshape(3, 640, 480)

        transform = torchvision.transforms.Resize((96, 96), #Get reshape numbers from group 3 and change them
              interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        
        color_image = transform(color_image)
        depth_image = transform(depth_image)

        gripper_pos = SawyerEnv.get_bariflex_state(self)

        return  color_image, depth_image, gripper_pos

    def get_starting_arrays(self):
        color_image, depth_image, gripper_pos = self.get_current_state()
        color_image_array, depth_image_array, gripper_pos_array = [], [], []

        for i in range(self.obs_horizon):
            color_image_array.append(color_image)
            depth_image_array.append(depth_image)
            gripper_pos_array.append(float(gripper_pos))

        return color_image_array, depth_image_array, gripper_pos_array

    def run(self, num_passes=2):
        colorimg_arr, depthimg_arr, gripperpos_arr = self.get_starting_arrays()

        noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
            )
        
        for i in range(num_passes):
            colorimg_tensor = torch.stack(colorimg_arr, dim=0).unsqueeze(0).squeeze(2)
            depthimg_tensor = torch.stack(depthimg_arr, dim=0).unsqueeze(0).squeeze(2)
            gripperpos_tensor = torch.stack(gripperpos_arr, dim=0).unsqueeze(0).squeeze(2)

            colorimg_tensor = dataset.normalize(colorimg_tensor, self.pose_stats).to(self.device)
            depthimg_tensor = dataset.normalize(depthimg_tensor, self.orientation_stats).to(self.device)
            gripperpos_tensor = dataset.normalize(gripperpos_tensor, self.image_stats).to(self.device)


            new_color_image = colorimg_tensor.reshape(1, self.obs_horizon, 3, 96, 96)
            new_depth_image = depthimg_tensor.reshape(1, self.obs_horizon, 3, 96, 96)
            agent_pos = torch.cat((gripperpos_tensor), dim=2).reshape(1, self.obs_horizon, 7)
            B = agent_pos.shape[0]

            color_image_features = self.nets["vision_encoder"](new_color_image.flatten(end_dim=1))
            color_image_features = color_image_features.reshape(*new_color_image.shape[:2],-1)
            depth_image_features = self.nets["vision_encoder"](new_depth_image.flatten(end_dim=1))
            depth_image_features = depth_image_features.reshape(*new_depth_image.shape[:2],-1)
            obs_features = torch.cat([color_image_features, depth_image_features, agent_pos], dim=-1)
            obs_cond = obs_features.flatten(start_dim=1)

            noisy_action = torch.randn((1, self.pred_horizon, 7), device=self.device)
            noise_scheduler.set_timesteps(100)

            for k in noise_scheduler.timesteps:
                noise_pred = self.nets["noise_pred_net"](
                    sample=noisy_action,
                    timestep=k,
                    global_cond=obs_cond
                )

                noisy_action = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=noisy_action
                ).prev_sample

            noisy_action = noisy_action.detach().to('cpu')
            noisy_action = noisy_action[0]
            # (1, 4, 7)

            (pred_pos, pred_or) = torch.split(noisy_action, [3, 4], dim=1)


            for j in range(self.pred_horizon):
                current_pos = dataset.unnormalize(pred_pos[j], self.pose_stats)
                current_or = dataset.unnormalize(pred_or[j], self.orientation_stats)

                try:
                    self.env.step([current_pos[0].item(), current_pos[1].item(), 
                        current_pos[2].item(), current_or[0].item(), 
                        current_or[1].item(), current_or[2].item(), 
                        current_or[3].item()])

                    colorimg_arr.pop(0)
                    depthimg_arr.pop(0)
                    gripperpos_arr.pop(0)

                    new_colorimg, new_depthimg, new_gripperpos = self.get_current_state()

                    colorimg_arr.append(new_colorimg)
                    depthimg_arr.append(new_depthimg)
                    gripperpos_arr.append(new_gripperpos)
                except:
                    continue
                


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    runner = Runner(device=device)

    runner.run(num_passes=1)
