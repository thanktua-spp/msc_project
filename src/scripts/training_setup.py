import os

import torch
import wandb
from diffusers import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
import scripts
import importlib
importlib.reload(scripts)
from scripts.model_config import TrainingConfig

def setup_training(nets, dataloader, cfg: TrainingConfig):
    """
    Initialize all training components based on configuration
    
    Args:
        nets: Dictionary containing model networks
        noise_pred_net: Noise prediction network
        dataloader: Training dataloader
        cfg: Training configuration
        
    Returns:
        Dictionary containing all training components
    """
    
    # Set device and move models
    device = torch.device(cfg.device)
    print(device)
    nets.to(device)
    
    # Initialize diffusion scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg.num_diffusion_iters,
        beta_schedule=cfg.beta_schedule,
        clip_sample=cfg.clip_sample,
        prediction_type=cfg.prediction_type
    )
    
    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(
        parameters=nets.parameters(),
        power=cfg.ema_power
    )
    
    # Standard ADAM optimizer
    # Note that EMA parameters are not optimized
    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=cfg.learning_rate, 
        weight_decay=cfg.weight_decay
    )
    
    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name=cfg.lr_scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=len(dataloader) * cfg.num_epochs
    )
    
    # Enhanced wandb initialization with comprehensive config
    if os.getenv("WANDB_MODE") != "disabled":
        config_dict = cfg.to_wandb_config(dataloader, nets)
        wandb.init(
            project=cfg.wandb_project,
            name=f"{config_dict['vision_encoder']}_{config_dict['noise_pred_net']}_exp_{wandb.util.generate_id()}", # Unique run name
            config=config_dict,
            tags=["diffusion-policy", "pusht", "imitation-learning"],
            notes="Training diffusion policy on PushT task with vision encoder",
            reinit=True
        )
    else:
        print("W&B logging disabled in setup_training")
    
    
    return {
        'device': device,
        'noise_scheduler': noise_scheduler,
        'ema': ema,
        'optimizer': optimizer,
        'lr_scheduler': lr_scheduler
    }