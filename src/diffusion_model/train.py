import dataset
import model

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

class DepthCNN(nn.Module):
    def __init__(self):
        super(DepthCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 64 * 64, 512)  # Adjust the dimensions based on your depth image size

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = nn.ReLU()(self.conv3(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Class to handle training model, saves important data
class Trainer():
    def __init__(self, demonstrations, obs_horizon, pred_horizon, 
                 pose_stats, orientation_stats, 
                 img_stats=None, batch_size=64, device="cpu", 
                 learning_rate=1e-4, save_file="checkpoint.pth"):
        '''
        Init function, saves relevant data in class.
        '''
        self.demos = demonstrations
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.pose_stats = pose_stats
        self.orientation_stats = orientation_stats
        self.img_stats = img_stats
        self.batch_size = batch_size
        self.device = device
        self.lr = learning_rate
        self.save_file = save_file

    def get_dataloader(self):
        '''
        Gets the dataloader for the training set.
        '''
        diff_dataset = dataset.DiffDataset(self.demos, self.obs_horizon, self.pred_horizon, self.pose_stats, 
                                           self.orientation_stats, self.img_stats)
        return torch.utils.data.DataLoader(diff_dataset, batch_size=self.batch_size, shuffle=True)

    def get_model(self):
        '''
        Gets model and exponential moving average.
        '''
        vision_encoder = model.get_resnet("resnet18")
        vision_encoder = model.replace_bn_with_gn(vision_encoder)

        depth_cnn = DepthCNN()

        vision_feature_dim = 512
        depth_feature_dim = 512
        lowdim_obs_dim = 7
        obs_dim = vision_feature_dim + depth_feature_dim + lowdim_obs_dim
        action_dim = 8

        noise_pred_net = model.ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_dim * self.obs_horizon
        )

        nets = nn.ModuleDict({
            'vision_encoder': vision_encoder,
            'depth_cnn': depth_cnn,
            'noise_pred_net': noise_pred_net
        })

        nets = nets.to(torch.device(self.device))

        ema = EMAModel(
            parameters=nets.parameters(),
            power=0.75
        )

        return nets, ema
    
    def get_noise_scheduler(self, num_diffusion_iters):
        '''
        Gets noise scheduler.
        '''
        return DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
    
    def train(self, checkpoint=None, num_epochs=20, print_stats=True):
        '''
        Runs training loop.
        '''
        num_diffusion_iters = 100

        min_loss = 1000000000
        min_epoch = 0

        trainloader = self.get_dataloader()
        nets, ema = self.get_model() if not checkpoint else model.load_pretrained(checkpoint, self.device, self.obs_horizon)
        noise_scheduler = self.get_noise_scheduler(num_diffusion_iters)

        optimizer = torch.optim.AdamW(
            params=nets.parameters(),
            lr=self.lr,
            weight_decay=1e-6
        )

        lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=len(trainloader) * num_epochs
        )

        epoch_loss = []

        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch in trainloader:
                
                rgb_image = batch["image_color"].to(self.device)
                depth_image = batch["image_depth"].to(self.device)
                agent_pos = batch["gripper_pos"].to(self.device)
                action = batch["actions"].to(self.device)
                B = agent_pos.shape[0]

                image_features = nets["vision_encoder"](rgb_image.flatten(end_dim=1))
                image_features = image_features.reshape(*rgb_image.shape[:2], -1)
                
                depth_features = nets["depth_cnn"](depth_image.flatten(end_dim=1))
                depth_features = depth_features.reshape(*depth_image.shape[:2], -1)
                
                obs_features = torch.cat([image_features, depth_features, agent_pos], dim=-1)
                obs_cond = obs_features.flatten(start_dim=1)

                noise = torch.randn(action.shape, device=self.device)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (B, ), device=self.device
                ).long()

                noisy_actions = noise_scheduler.add_noise(
                    action, noise, timesteps)
                
                noise_pred = nets["noise_pred_net"](noisy_actions, 
                                                    timesteps, global_cond=obs_cond)
                
                loss = nn.functional.mse_loss(noise_pred, noise)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                lr_scheduler.step()
                ema.step(nets.parameters())

                loss_cpu = loss.item()
                total_loss += loss_cpu
            
            if print_stats:
                print(f"Epoch {epoch + 1}: Loss: {total_loss}")

            if total_loss < min_loss:
                    min_loss = total_loss
                    min_epoch = epoch + 1

        torch.save(nets.state_dict(), self.save_file)
        
        ema_nets = nets
        ema.copy_to(ema_nets.parameters())

        print(f"\n\nMinimum Loss: {min_loss:.4f} on epoch {min_epoch}")
    

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    dataset_file = "../../test_cup_grabbing.hdf5"

    (dataset_list, pos_stats, or_stats) = get_into_dataloader_format(dataset_file)

    with open("stats.json", "r") as file:
        stats_dict = json.load(file)
        stats_dict["pose_stats"] = pos_stats
        stats_dict["orientation_stats"] = or_stats

    with open("stats.json", "w") as file:
        json.dump(stats_dict, file, indent=4)

    pred_horizon = 4
    obs_horizon = 6

    print(dataset_list[0][0]["rgb_image"].shape)
    print(dataset_list[0][0]["position"].shape)
    print(dataset_list[0][0]["orientation"].shape)

    trainer = Trainer(dataset_list, obs_horizon, pred_horizon, 
        pos_stats, or_stats, batch_size=64, device=device, learning_rate=5e-6)

    dataloader = trainer.get_dataloader()
    print(f"Length of Dataloader: {len(dataloader)}")
    for batch in dataloader:
        print(batch["rgb_image"].shape)
        print(batch["agent_pos"].shape)
        print(batch["action"].shape)
        break

    print("\nTraining...\n")
    trainer.train(checkpoint=None, num_epochs=200, print_stats=True)
