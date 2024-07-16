import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import h5py
import cv2


#Padded Observations
def get_pad_obs(demo, num_times_zero, index, color_stats, depth_stats, gripper_stats):

    temp_color_list = np.array([])
    temp_depth_list = np.array([])
    temp_grip_list = np.array([])

    #Pad first instance x num of times until obs_horizon
    for i in range(num_times_zero):
        
        color = demo["obs/color"][0]
        color = normalize(torch.from_numpy(color).float(), color_stats).permute(2, 0, 1)
        temp_color_list = np.append(temp_color_list, color)
    
        depth = demo["obs/depth"][0]
        depth = normalize(torch.from_numpy(depth).float(), depth_stats).unsqueeze(2).permute(2, 0, 1)
        temp_depth_list = np.append(temp_depth_list, depth)

        grip = demo["obs/pos"][0]
        grip = normalize(grip, gripper_stats)
        temp_grip_list = np.append(temp_grip_list, grip)

    idx = 1

    while idx < index + 1:

        color = demo["obs/color"][idx]
        color = normalize(torch.from_numpy(color).float(), color_stats).permute(2, 0, 1)
        temp_color_list = np.append(temp_color_list, color)

        depth = demo["obs/depth"][idx]
        depth = normalize(torch.from_numpy(depth).float(), depth_stats).unsqueeze(2).permute(2, 0, 1)
        temp_depth_list = np.append(temp_depth_list, depth)

        grip = demo["obs/pos"][idx]
        grip = normalize(grip, gripper_stats)
        temp_grip_list = np.append(temp_grip_list, grip)

        idx += 1
    
    # transform = transforms.Resize((96, 96), interpolation=transforms.InterpolationMode.BILINEAR)
    
    
    color_obs = cv2.resize(temp_color_list, (96, 96), interpolation= cv2.INTER_LINEAR)
    depth_obs = cv2.resize(temp_depth_list, (96, 96), interpolation= cv2.INTER_LINEAR)
    gripper_obs = torch.FloatTensor(temp_grip_list)

    return color_obs, depth_obs, gripper_obs


#Non-Padded Observations
def get_obs(demo, start_idx, index, color_stats, depth_stats, gripper_stats):

    temp_color_list = []
    temp_depth_list = []
    temp_grip_list = []
    
    
    idx = start_idx

    while idx < index + 1:

        color = demo["obs/color"][idx]
        color = normalize(torch.from_numpy(color).float(), color_stats).permute(2, 0, 1)
        temp_color_list = np.append(temp_color_list, color)

        depth = demo["obs/depth"][idx]
        depth = normalize(torch.from_numpy(depth).float(), depth_stats).unsqueeze(2).permute(2, 0, 1)
        temp_depth_list = np.append(temp_depth_list, depth)

        grip = demo["obs/pos"][idx]
        grip = normalize(grip, gripper_stats)
        temp_grip_list = np.append(temp_grip_list, grip)

        idx += 1

    transform = transforms.Resize((96, 96), interpolation=transforms.InterpolationMode.BILINEAR)

    color_obs = cv2.resize(temp_color_list, (96, 96), interpolation= cv2.INTER_LINEAR)
    depth_obs = cv2.resize(temp_depth_list, (96, 96), interpolation= cv2.INTER_LINEAR)
    gripper_obs = torch.FloatTensor(temp_grip_list)

    return color_obs, depth_obs, gripper_obs
    

#Get Actions
def get_action(demo, pred_horizon, start_idx):
    count = 0
    idx = start_idx
    temp_action_list = []

    while count < pred_horizon and idx < demo["num_samples"][()]:
        temp_action_list.append(demo["actions"][idx][()])
        count += 1
        idx += 1

    while count < pred_horizon:
        temp_action_list.append(demo["actions"][len(demo["actions"][()]) - 1])
        count += 1
    
    return temp_action_list


#Normalize Stats
def normalize(tensor, stats):
    
    return ((tensor - stats["min"]) / stats["max"]) * 2 - 1


def get_stats(tensor):
    stats = {}
    stats["min"] = torch.min(tensor)
    stats["max"] = torch.max(tensor)

    return stats


def unnormalize(tensor, stats):

    return ((tensor + 1) / 2) * stats["max"] + stats["min"]


#Main Class
class DiffDataset(Dataset):
    def __init__(self, demonstrations, obs_horizon, pred_horizon, color_stats, depth_stats, gripper_stats):
        self.dataset = []
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.color_stats = color_stats
        self.depth_stats = depth_stats
        self.gripper_stats = gripper_stats


        #Define stats
        if (color_stats is None):
            self.color_stats = {"min" : 0, "max" : 255}
        else:
            self.color_stats = get_stats(color_stats)

        if (depth_stats is None):
            self.depth_stats = {"min" : 0, "max" : 255}
        else:
            self.depth_stats = get_stats(depth_stats)

        if (gripper_stats is None):
            self.gripper_stats = {"min" : -1, "max" : 1}
        else:
            self.gripper_stats = get_stats(gripper_stats)    

        #Obs horizon padding
        for demo in demonstrations:
            for i in range(demonstrations[demo]["num_samples"][()]):
               
                print("index: ", i, "demo: ", demo)
                print("running")
               
                temp = {}
                start_idx = i - self.obs_horizon + 1
                num_times_zero = 0
                if (start_idx == 0):
                    num_times_zero = 1
                elif (start_idx < 0):
                    num_times_zero = abs(start_idx) + 1

                if (num_times_zero > 0):
                    color_obs, depth_obs, gripper_obs = get_pad_obs(demonstrations[demo],
                                        num_times_zero, i,
                                        self.color_stats,
                                        self.depth_stats,
                                        self.gripper_stats)
                else:
                    color_obs, depth_obs, gripper_obs = get_obs(demonstrations[demo],
                                        start_idx, i,
                                        self.color_stats,
                                        self.depth_stats,
                                        self.gripper_stats)
        
                
                action_pred = get_action(demonstrations[demo],
                                pred_horizon, i + 1)
                
                temp["actions"] = action_pred
                temp["color"] = color_obs
                temp["depth"] = depth_obs
                temp["gripper_state"] = gripper_obs
                self.dataset.append(temp)
    
    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        nsample = {}

        temp = self.dataset[idx]

        nsample["color"] = (temp["color"])
        nsample["depth"] = (temp["depth"])
        nsample["gripper_pos"] = temp["gripper_state"].squeeze(1).float()
        nsample["actions"] = temp["actions"].squeeze(1).float()
        return nsample


#Runner
if __name__ == '__main__':
    name = "Summer-Camp-Rosbag/scripts/rosbag.hdf5" #File name as str
    file = h5py.File(name, 'r') #File reader

    obs_horizon = 6
    pred_horizon = 4

    color_stats = None
    depth_stats = None
    gripper_stats = None

    dataset = DiffDataset(file, obs_horizon, pred_horizon, color_stats, depth_stats, gripper_stats)

