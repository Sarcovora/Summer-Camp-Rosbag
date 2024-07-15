# file imports
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

import os
import hydra
from pprint import pprint 


from helper_functions import get_into_dataloader_format

CONFIG = os.path.join(os.getcwd(), 'config')

# Class to handle training model, saves important data
class Trainer():
    def __init__(self, 
                 cfg,
                 demonstrations,
                 pose_stats,
                 orientation_stats,
                 img_stats=None
                #  demonstrations, 
                #  obs_horizon, 
                #  pred_horizon, 
                #  pose_stats, 
                #  orientation_stats, 
                #  img_stats=None, 
                #  batch_size=64, 
                #  device="cpu", 
                #  learning_rate=1e-4, 
                #  save_file="checkpoint.pth"):
    ):
        '''
        Init function, saves relevant data in class.
        '''
        self.device = cfg.train.device

        self.demos = demonstrations
        self.pose_stats = pose_stats
        self.orientation_stats = orientation_stats
        self.img_stats = img_stats
        
        self.obs_keys = cfg.obs.obs_keys
        self.obs_horizon = cfg.model.obs_horizon
        self.pred_horizon = cfg.model.pred_horizon
        self.execution_horizon = cfg.model.execution_horizon
         
        self.vision_feature_dim = cfg.model.vision_encoder_kwargs.vision_feature_dim
        self.lowdim_obs_dim = cfg.model.vision_encoder_kwargs.lowdim_obs_dim
        self.obs_dim = self.vision_feature_dim * 2 + self.lowdim_obs_dim
        self.action_dim = cfg.model.vision_encoder_kwargs.action_dim
        self.num_diffusion_iters = cfg.train.scheduler.noise_scheduler.num_diffusion_iters
        
        self.batch_size = cfg.train.batch_size
        self.lr = cfg.train.optim.learning_rate
        self.weight_decay = cfg.train.optim.weight_decay
        self.num_epochs = cfg.train.num_epochs
        
        self.lr_scheduler = cfg.train.scheduler.lr_scheduler.name
        self.num_wramup_steps = cfg.train.scheduler.lr_scheduler.num_warmup_steps    

        self.save_file = cfg.train.save_file
        
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
        
        depth_encoder = model.get_cnn("nature_cnn")

        noise_pred_net = model.ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.obs_dim*self.obs_horizon
        )

        nets = nn.ModuleDict({
            'vision_encoder': vision_encoder,
            'depth_encoder': depth_encoder,
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
    
    def train(self, checkpoint=None, print_stats=True):
        '''
        Runs training loop.
        '''

        min_loss = 1000000000
        min_epoch = 0

        trainloader = self.get_dataloader()
        nets, ema = self.get_model() if not checkpoint else model.load_pretrained(checkpoint, self.device, self.obs_horizon)
        noise_scheduler = self.get_noise_scheduler(self.num_diffusion_iters)

        optimizer = torch.optim.AdamW(
            params=nets.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        lr_scheduler = get_scheduler(
            name=self.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.num_wramup_steps,
            num_training_steps=len(trainloader) * self.num_epochs
        )

        epoch_loss = []

        for epoch in range(self.num_epochs):
            total_loss = 0.0
            for batch in trainloader:
                image, depth_image, agent_pos, action = [batch[key].to(self.device) for key in self.obs_keys]
                B = agent_pos.shape[0]

                image_features = nets["vision_encoder"](image.flatten(end_dim=1))
                image_features = image_features.reshape(*image.shape[:2],-1)                
                
                # cnn (depth image)
                depth_features = nets["depth_encoder"](depth_image.flatten(end_dim=1)).reshape(*depth_image.shape[:2],-1)
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
    

@hydra.main(config_path=CONFIG, config_name="config", version_base="1.3.2")
def main(cfg):
    pprint(cfg)

    # (dataset_list, pos_stats, or_stats) = get_into_dataloader_format(dataset_file)

    # with open("stats.json", "r") as file:
    #     stats_dict = json.load(file)
    #     stats_dict["pose_stats"] = pos_stats
    #     stats_dict["orientation_stats"] = or_stats

    # with open("stats.json", "w") as file:
    #     json.dump(stats_dict, file, indent=4)


    pred_horizon = 4
    obs_horizon = 6

    # print(dataset_list[0][0]["image"].shape)
    # print(dataset_list[0][0]["position"].shape)
    # print(dataset_list[0][0]["orientation"].shape)

    trainer = Trainer(cfg, None, None, None)
    
    nets, ema = trainer.get_model()
    
    image = torch.zeros((1, obs_horizon, 3, 96, 96))
    depth_image = torch.zeros((1, obs_horizon, 1, 96, 96))
    agent_pos = torch.zeros((1, obs_horizon, 1))    # gripper state
    action = torch.zeros((1, pred_horizon, 8))
    
    B = agent_pos.shape[0]

    image_features = nets["vision_encoder"](image.flatten(end_dim=1))
    image_features = image_features.reshape(*image.shape[:2],-1)
    
    depth_features = nets["depth_encoder"](depth_image.flatten(end_dim=1)).reshape(*depth_image.shape[:2],-1)
    
    obs_features = torch.cat([image_features, depth_features, agent_pos], dim=-1)
    obs_cond = obs_features.flatten(start_dim=1)
    
    noisy_actions = torch.randn(action.shape)
    diffusion_iter = torch.zeros((1,))
    
    noise_pred = nets["noise_pred_net"](noisy_actions, 
                                        diffusion_iter, global_cond=obs_cond)
    
    denoised_actions = noise_pred - noise_pred
    print(denoised_actions.shape, denoised_actions)
    

if __name__ == "__main__":
    main()