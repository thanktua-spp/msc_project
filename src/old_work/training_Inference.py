import os
from pathlib import Path
import numpy as np
import wandb
from tqdm import tqdm

import torch
import torch.nn as nn
import scripts

import importlib
importlib.reload(scripts)
from scripts.model_config import TrainingConfig
from scripts.training_setup import setup_training


class UnetTrainer:
    def __init__(self, nets, dataloader):
        self.nets = nets
        self.dataloader = dataloader
        
    def train_diffusion_policy(self, epochs, save=False, save_to=''):
        """
        Main training function using the configuration structure
        """
        
        # Initialize configuration
        cfg = TrainingConfig(
            num_epochs=epochs,
            #obs_horizon=obs_horizon,
            # You can override any default values here
            # pred_horizon=your_pred_horizon,
            # action_horizon=your_action_horizon,
        )
        
        # Setup all training components
        training_components = setup_training(self.nets, self.dataloader, cfg)
        
        device = training_components['device']
        noise_scheduler = training_components['noise_scheduler']
        ema = training_components['ema']
        optimizer = training_components['optimizer']
        lr_scheduler = training_components['lr_scheduler']
        
        # Track model architecture
        #wandb.watch(nets, log="all", log_freq=100)
            
        # Training loop with enhanced logging
        with tqdm(range(cfg.num_epochs), desc='Epoch') as tglobal:
            for epoch_idx in tglobal:
                epoch_loss = []
                epoch_lr = []
                epoch_timesteps = []
                
                # Batch loop with detailed logging
                with tqdm(self.dataloader, desc='Batch', leave=False) as tepoch:
                    for batch_idx, nbatch in enumerate(tepoch):
                        # Your existing training code...
                        nimage = nbatch['image'][:,:cfg.obs_horizon].to(device)
                        nagent_pos = nbatch['agent_pos'][:,:cfg.obs_horizon].to(device)
                        naction = nbatch['action'].to(device)
                        B = nagent_pos.shape[0]

                        # Encoder vision features
                        image_features = self.nets['vision_encoder'](
                            nimage.flatten(end_dim=1))
                        image_features = image_features.reshape(*nimage.shape[:2], -1)
                        
                        # Concatenate vision feature and low-dim obs
                        obs_features = torch.cat([image_features, nagent_pos], dim=-1)
                        obs_cond = obs_features.flatten(start_dim=1)
                        
                        # Sample noise and timesteps
                        noise = torch.randn(naction.shape, device=device)
                        timesteps = torch.randint(
                            0, noise_scheduler.config.num_train_timesteps,
                            (B,), device=device
                        ).long()
                            
                        # Forward diffusion process
                        noisy_actions = noise_scheduler.add_noise(
                            naction, noise, timesteps)
                        
                        # Predict noise
                        noise_pred = self.nets['noise_pred_net'](
                            noisy_actions, timesteps, global_cond=obs_cond)
                        
                        # Compute loss
                        loss = nn.functional.mse_loss(noise_pred, noise)
                        
                        # Optimization
                        loss.backward()
                        
                        # Track gradient norms (useful for debugging)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.nets.parameters(), max_norm=float('inf'))
                        
                        optimizer.step()
                        optimizer.zero_grad()
                        lr_scheduler.step()
                        ema.step(self.nets.parameters())
                        
                        # Collect metrics
                        loss_cpu = loss.item()
                        current_lr = lr_scheduler.get_last_lr()[0]
                        mean_timestep = timesteps.float().mean().item()
                        
                        epoch_loss.append(loss_cpu)
                        epoch_lr.append(current_lr)
                        epoch_timesteps.append(mean_timestep)
                        
                        # Log every N steps (avoid overwhelming wandb)
                        global_step = epoch_idx * len(self.dataloader) + batch_idx
                        if batch_idx % cfg.log_every_n_batches == 0:
                            wandb.log({
                                "train/batch_loss": loss_cpu,
                                "train/learning_rate": current_lr,
                                "train/grad_norm": grad_norm.item(),
                                "train/mean_timestep": mean_timestep,
                                "train/epoch": epoch_idx,
                                "train/global_step": global_step,
                            })
                        
                        # Update progress bars
                        tepoch.set_postfix(loss=loss_cpu, lr=f"{current_lr:.2e}")
                        tglobal.set_postfix(loss=np.mean(epoch_loss))
                
                # End of epoch logging
                epoch_metrics = {
                    "epoch/avg_loss": np.mean(epoch_loss),
                    "epoch/min_loss": np.min(epoch_loss),
                    "epoch/max_loss": np.max(epoch_loss),
                    "epoch/std_loss": np.std(epoch_loss),
                    "epoch/final_lr": epoch_lr[-1],
                    "epoch/avg_timestep": np.mean(epoch_timesteps),
                    "epoch/number": epoch_idx,
                }
                
                wandb.log(epoch_metrics)
                
                #print(f"Epoch {epoch_idx}: Avg Loss = {epoch_metrics['epoch/avg_loss']:.6f}")

        # Final model logging
        wandb.log({
            "training/total_epochs": cfg.num_epochs,
            "training/final_loss": epoch_loss[-1],
        })

        # Save EMA model
        ema_nets = self.nets
        ema.copy_to(ema_nets.parameters())

        # Optional: Save model artifacts
        checkpoint_name = str(save_to) + f"{cfg.num_epochs}_model_checkpoint.pth"
        if save:
            torch.save(self.nets.state_dict(), checkpoint_name)

            # Use the same filename for wandb
            wandb_save = f"model_checkpoint.pth"
            wandb.save(wandb_save)
            artifact = wandb.Artifact("diffusion_policy_model", type="model")
            artifact.add_file(wandb_save)
            wandb.log_artifact(artifact)

        wandb.finish()
        print("Training completed and logged to wandb!")
        return ema_nets


    # inference code