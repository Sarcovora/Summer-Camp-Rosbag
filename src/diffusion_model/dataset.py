# imports 
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np


# helper functions

def get_pad_obs(demo, num_times_zero, index, color_stats, depth_stats, gripper_stats):
  '''
  Returns observation data when padding is needed at the beginning.

  '''
  temp_color_list = []
  temp_depth_list = []
  temp_gripper_pos_list = []
  for x in range(num_times_zero):
    # pads the necessary amount of times
    color = torch.from_numpy(demo[0]["image_color"])
    color = normalize(color, color_stats)
    
    temp_color_list.append(color)

    depth = torch.from_numpy(demo[0]["image_depth"])
    depth = normalize(depth, depth_stats)
    temp_depth_list.append(depth)
    

    gripper_pos = torch.from_numpy(demo[0]["gripper_pos"])

    temp_grip = normalize(temp_grip, gripper_stats)
    temp_gripper_pos_list.append(gripper_pos)

  idx = 1

  while idx < index + 1:
    # remaining observations
    color = torch.from_numpy(demo[idx]["image_color"])
    color = normalize(color, color_stats)
    
    depth = torch.from_numpy(demo[idx]["image_depth"])
    depth = normalize(depth, depth_stats)
    temp_depth_list.append(depth)
    
    temp_color_list.append(color)
    temp_grip = torch.from_numpy(demo[idx]["gripper_pos"])
    temp_grip = normalize(temp_grip, gripper_stats)
  

    temp_gripper_pos_list.append(gripper_pos)

    idx += 1

  # returns data as torch tensors
  depth_obs = torch.stack(temp_depth_list)
  transform = transforms.Resize((96, 96),
              interpolation=transforms.InterpolationMode.BILINEAR)
  depth_obs = transform(depth_obs)
  gripper_obs = torch.stack(temp_gripper_pos_list)
  color_obs = torch.stack(temp_color_list)
  color_obs = transform(color_obs)


  return color_obs, depth_obs, gripper_obs



def get_obs(demo, start_idx, index, color_stats, depth_stats, gripper_stats):
  '''
  Returns observation data when padding is not needed. 
  '''
  temp_color_list = []
  temp_depth_list = []
  temp_gripper_pos_list = []

  idx = start_idx

  while idx < index + 1:
    color = torch.from_numpy(demo[idx]["image_color"])
    color = normalize(color, color_stats)

    temp_color_list.append(color)
    temp_gripper_pos = torch.from_numpy(demo[idx]["gripper_pos"])


    temp_gripper_pos = normalize(temp_gripper_pos_list, gripper_stats)
    temp_gripper_pos_list.append(temp_gripper_pos)

    idx += 1

  # returns data as torch tensors  
  depth_obs = torch.stack(temp_depth_list)
  transform = transforms.Resize((96, 96),
              interpolation=transforms.InterpolationMode.BILINEAR)
  depth_obs = transform(depth_obs)
  gripper_obs = torch.stack(temp_gripper_pos_list)
  color_obs = torch.stack(temp_color_list)
  color_obs = transform(color_obs)


  return color_obs, depth_obs, gripper_obs



def get_action(demo, pred_horizon, start_idx):
  '''
  Returns action data, handles end padding if needed. 
  '''
  count = 0
  index = start_idx

  temp_action_list = []


  demo[index]["actions"] 
  demo[index+pred_horizon]["actions"]
  return # (pred_horizon, 8)
   # ----

  while count < pred_horizon and index < len(demo):
    # gets actions until pred_horizon is satisfied or end of list is reached.
   




   

    temp_action_list.append(demo[index]["actions"])

    count += 1
    index += 1

  while count < pred_horizon:
    # handles padding if needed. 

    temp_action_list.append(temp_action_list[len(demo)-1]["actions"])

    count += 1

  # returns data as tensor  
  return torch.stack(temp_action_list)



def normalize(tensor, stats):
  '''
  Normalizes values given min and max to be between -1 and 1
  '''
  return ((tensor - stats["min"]) / stats["max"]) * 2 - 1



def get_stats(tensor):
  '''
  Returns the stats (containing min and max) of a given tensor
  '''
  stats = {}
  stats["min"] = torch.min(tensor)
  stats["max"] = torch.max(tensor)

  return stats



def unnormalize(tensor, stats):
  '''
  Unnormalizes a tensor given the stats. Will be used for inference. 
  '''
  return ((tensor + 1) / 2) * stats["max"] + stats["min"]


# Dataset class. Returns data as shown below
# F : tensor size (batch_size, obs_horizon, 3, 96, 96)
# agent_pos : tensor size (batch_size, obs_horizon, 7)
# action: tensor size (batch_size, pred_horizon, 7)
class DiffDataset(Dataset):
  def __init__(self, demonstrations, obs_horizon, pred_horizon, 
               color_stats, depth_stats, gripper_stats):
    '''
    Initializes dataset.
    '''
    self.dataset = []
    self.obs_horizon = obs_horizon
    self.pred_horizon = pred_horizon
    self.color_stats = color_stats
    self.depth_stats = depth_stats
    self.gripper_stats = gripper_stats

    if (color_stats == None):
      self.color_stats = {"min" : 0, "max" : 255}
    else:
      self.color_stats = color_stats

    if (depth_stats == None):
      self.depth_stats = {"min" : 0, "max" : 255} #not sure if depth image should be compressed to 255
    else:
      self.depth_stats = depth_stats

    for demo in demonstrations:
      for i in range(len(demo) - 1):
        temp = {}
        start_idx = i - self.obs_horizon + 1
        # gets the number of times the zero indexed obs is in the observation list
        num_times_zero = 0
        if (start_idx == 0):
          num_times_zero = 1
        elif (start_idx < 0):
          num_times_zero = abs(start_idx) + 1



        # getting observations
        if (num_times_zero > 0):
          color_obs, depth_obs, gripper_obs = get_pad_obs(demo,
                                            num_times_zero, i, 
                                            self.color_stats, 
                                            self.depth_stats, 
                                            self.gripper_stats)
        else:
          color_obs, depth_obs, gripper_obs(demo, 
                                         start_idx, i, 
                                         self.color_stats, 
                                         self.depth_stats,
                                         self.gripper_stats)
          
        # getting actions
        action_pred = get_action(demo, 
                                 pred_horizon, i + 1,)

        temp["image_color"] = color_obs
        temp["image_depth"] = depth_obs
        temp["gripper_pos"] = gripper_obs
        temp["actions"] = action_pred
        self.dataset.append(temp)

  def __len__(self):
    '''
    Returns length of dataset.
    '''
    return len(self.dataset)

  def __getitem__(self, idx):
    '''
    Gets datapoint at specified index.
    '''
    nsample = {}

    temp = self.dataset[idx]

    nsample["color"] = (temp["color"])
    nsample["depth"] = (temp["depth"])
    nsample["gripper_pos"] = temp["gripper_pos"].squeeze(1).float()
    nsample["actions"] = temp["actions"].squeeze(1).float()

    return nsample
