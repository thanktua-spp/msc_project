import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class TrainingConfig:
    # Training hyperparameters
    num_epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    batch_size: int = 64
    
    # Optimizer settings
    optimizer_name: str = "AdamW"
    
    # Learning rate scheduler
    lr_scheduler_name: str = "cosine"
    warmup_steps: int = 500
    
    # EMA parameters
    ema_power: float = 0.75
    
    # Diffusion scheduler parameters
    num_diffusion_iters: int = 100
    beta_schedule: str = 'squaredcos_cap_v2'
    clip_sample: bool = True
    prediction_type: str = 'epsilon'
    
    # Model architecture (will be set based on your specific setup)
    obs_horizon: Optional[int] = 2
    pred_horizon: Optional[int] = 8
    action_horizon: Optional[int] = 16
    
    # Logging settings
    wandb_project: str = "diffusion-policy-pusht"
    log_every_n_batches: int = 10
    
    # Device settings
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def to_wandb_config(self, dataloader, nets) -> Dict[str, Any]:
        """Convert config to wandb-compatible dictionary with additional runtime info"""
        config_dict = {
            # Training hyperparameters
            "epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "optimizer": self.optimizer_name,
            "lr_scheduler": self.lr_scheduler_name,
            "warmup_steps": self.warmup_steps,
            "batch_size": getattr(dataloader, 'batch_size', self.batch_size),
            
            # Model architecture
            "obs_horizon": self.obs_horizon,
            "pred_horizon": self.pred_horizon,
            "action_horizon": self.action_horizon,
            "vision_encoder": str(nets['vision_encoder'].__class__.__name__).lower(),
            "noise_pred_net": str(nets['noise_pred_net'].__class__.__name__).lower(),
            
            # EMA parameters
            "ema_power": self.ema_power,
            
            # Diffusion scheduler parameters
            "num_diffusion_iters": self.num_diffusion_iters,
            "beta_schedule": self.beta_schedule,
            "clip_sample": self.clip_sample,
            "prediction_type": self.prediction_type,
            
            # Dataset info
            "dataset_size": len(dataloader.dataset) if hasattr(dataloader, 'dataset') else None,
            "num_batches": len(dataloader),
            
            # Device and hardware info
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
        }
        
        # Add GPU info if available
        if torch.cuda.is_available():
            config_dict.update({
                "gpu_name": torch.cuda.get_device_name(),
                "gpu_memory_total": torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            })
        
        return config_dict