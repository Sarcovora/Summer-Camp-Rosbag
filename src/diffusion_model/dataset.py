# imports 
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np


# helper functions

def get_pad_obs(demo, num_times_zero, index, pose_stats, or_stats, img_stats):
  '''
  Returns observation data when padding is needed at the beginning.
  '''
  temp_image_list = []
  temp_agent_pos_list = []
  for x in range(num_times_zero):
    # pads the necessary amount of times
    img = torch.from_numpy(demo[0]["image"])
    img = normalize(img, img_stats)
    
    temp_image_list.append(img)

    temp_pos = torch.from_numpy(demo[0]["position"])
    temp_or = torch.from_numpy(demo[0]["orientation"])

    temp_pos = normalize(temp_pos, pose_stats)
    temp_or = normalize(temp_or, or_stats)
    temp_agent_pos_list.append(torch.cat((temp_pos, temp_or), dim=1))

  idx = 1

  while idx < index + 1:
    # remaining observations
    img = torch.from_numpy(demo[idx]["image"])
    img = normalize(img, img_stats)
    
    temp_image_list.append(img)
    temp_pos = torch.from_numpy(demo[idx]["position"])
    temp_or = torch.from_numpy(demo[idx]["orientation"])

    temp_pos = normalize(temp_pos, pose_stats)
    temp_or = normalize(temp_or, or_stats)
    temp_agent_pos_list.append(torch.cat((temp_pos, temp_or), dim=1))

    idx += 1

  # returns data as torch tensors
  image_obs = torch.stack(temp_image_list)
  transform = transforms.Resize((96, 96),
              interpolation=transforms.InterpolationMode.BILINEAR)
  image_obs = transform(image_obs)
  agent_obs = torch.stack(temp_agent_pos_list)


  return image_obs, agent_obs



def get_obs(demo, start_idx, index, pose_stats, or_stats, img_stats):
  '''
  Returns observation data when padding is not needed. 
  '''
  temp_image_list = []
  temp_agent_pos_list = []

  idx = start_idx

  while idx < index + 1:
    img = torch.from_numpy(demo[idx]["image"])
    img = normalize(img, img_stats)

    temp_image_list.append(img)
    temp_pos = torch.from_numpy(demo[idx]["position"])
    temp_or = torch.from_numpy(demo[idx]["orientation"])

    temp_pos = normalize(temp_pos, pose_stats)
    temp_or = normalize(temp_or, or_stats)
    temp_agent_pos_list.append(torch.cat((temp_pos, temp_or), dim=1))

    idx += 1

  # returns data as torch tensors  
  image_obs = torch.stack(temp_image_list)
  transform = transforms.Resize((96, 96),
              interpolation=transforms.InterpolationMode.BILINEAR)
  image_obs = transform(image_obs)
  agent_obs = torch.stack(temp_agent_pos_list)


  return image_obs, agent_obs



def get_action(demo, pred_horizon, start_idx, pose_stats, or_stats):
  '''
  Returns action data, handles end padding if needed. 
  '''
  count = 0
  index = start_idx

  temp_action_list = []

  while count < pred_horizon and index < len(demo):
    # gets actions until pred_horizon is satisfied or end of list is reached.
    temp_pos = torch.from_numpy(demo[index]["position"])
    temp_or = torch.from_numpy(demo[index]["orientation"])

    temp_pos = normalize(temp_pos, pose_stats)
    temp_or = normalize(temp_or, or_stats)
    temp_action_list.append(torch.cat((temp_pos, temp_or), dim=1))

    count += 1
    index += 1

  while count < pred_horizon:
    # handles padding if needed. 
    temp_pos = torch.from_numpy(demo[len(demo) - 1]["position"])
    temp_or = torch.from_numpy(demo[len(demo) - 1]["orientation"])

    temp_pos = normalize(temp_pos, pose_stats)
    temp_or = normalize(temp_or, or_stats)
    temp_action_list.append(torch.cat((temp_pos, temp_or), dim=1))

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
# image : tensor size (batch_size, obs_horizon, 3, 96, 96)
# agent_pos : tensor size (batch_size, obs_horizon, 7)
# action: tensor size (batch_size, pred_horizon, 7)
class DiffDataset(Dataset):
  def __init__(self, demonstrations, obs_horizon, pred_horizon, 
               pose_stats, orientation_stats, img_stats=None):
    '''
    Initializes dataset.
    '''
    self.dataset = []
    self.obs_horizon = obs_horizon
    self.pred_horizon = pred_horizon
    self.pose_stats = pose_stats
    self.orientation_stats = orientation_stats

    if (img_stats == None):
      self.img_stats = {"min" : 0, "max" : 255}
    else:
      self.img_stats = img_stats


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

        temp_image_list = []
        temp_agent_pos_list = []

        # getting observations
        if (num_times_zero > 0):
          image_obs, agent_obs = get_pad_obs(demo,
                                            num_times_zero, i, 
                                            self.pose_stats, 
                                            self.orientation_stats, 
                                            self.img_stats)
        else:
          image_obs, agent_obs = get_obs(demo, 
                                         start_idx, i, 
                                         self.pose_stats, 
                                         self.orientation_stats,
                                         self.img_stats)
          
        # getting actions
        action_pred = get_action(demo, 
                                 pred_horizon, i + 1, 
                                 self.pose_stats, 
                                 self.orientation_stats)

        temp["image"] = image_obs
        temp["agent_pos"] = agent_obs
        temp["action"] = action_pred
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

    nsample["image"] = (temp["image"])
    nsample["agent_pos"] = temp["agent_pos"].squeeze(1).float()
    nsample["action"] = temp["action"].squeeze(1).float()

    return nsample
