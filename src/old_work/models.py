import sys
import os
import numpy as np
from pathlib import Path
import collections
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import math

import torchvision

class VisionEncoder:
    """
        Given a base classifier model
        1. Remove classification head
        2. Replace all BatchNorm layers with GroupNorm
    """
    def replace_submodules(
        self,
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]
    ) -> nn.Module:
        """Replace all submodules that match the predicate."""
        if predicate(root_module):
            return func(root_module)

        matches = [k.split('.') for k, m
            in root_module.named_modules(remove_duplicate=True)
            if predicate(m)]

        for *parent, k in matches:
            parent_module = root_module
            if parent:
                parent_module = root_module.get_submodule('.'.join(parent))

            src_module = parent_module[int(k)] if isinstance(parent_module, nn.Sequential) else getattr(parent_module, k)
            tgt_module = func(src_module)

            if isinstance(parent_module, nn.Sequential):
                parent_module[int(k)] = tgt_module
            else:
                setattr(parent_module, k, tgt_module)

        assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
        return root_module


    # --- Step 1: Replace last classifier layer with Identity ---
    def replace_last_classifier_with_identity(self, model: nn.Module) -> nn.Module:
        """Replace the last top-level child module with nn.Identity."""
        children_list = list(model.named_children())
        if not children_list:
            raise ValueError("Model has no child layers to replace.")
        last_name, _ = children_list[-1]
        setattr(model, last_name, nn.Identity())
        return model


    # --- Step 2: Replace BatchNorm with GroupNorm ---
    def replace_bn_with_gn(self, model: nn.Module, features_per_group: int = 16) -> nn.Module:
        """Replace all BatchNorm2d layers with GroupNorm, adjusting groups if needed."""
        def bn_to_gn(x: nn.BatchNorm2d) -> nn.GroupNorm:
            num_channels = x.num_features
            # Pick a valid number of groups
            num_groups = max(1, num_channels // features_per_group)
            # Ensure divisibility
            if num_channels % num_groups != 0:
                num_groups = math.gcd(num_channels, features_per_group) or 1
            return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)

        return self.replace_submodules(
            root_module=model,
            predicate=lambda m: isinstance(m, nn.BatchNorm2d),
            func=bn_to_gn
        )

    def get_model(self, model_name):
        model = torchvision.models.get_model(model_name, weights=None)
        model = self.replace_last_classifier_with_identity(model)
        model = self.replace_bn_with_gn(model)

        dummy_input = torch.zeros((1,3,1,1))
        emb_dim = model(dummy_input).shape[1]
        return model, emb_dim


class SinusoidalPosEmb(nn.Module):
    """
    dim: Dimension of time embedding 256
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: conditioning timestep (B,)
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class SinusoidalEncoder(nn.Module):
    """
    dsed: Dimension of time embedding (B, 256)
    """

    def __init__(self, dsed):
        super().__init__()
        self.pos_emb = SinusoidalPosEmb(dsed)

        self.mlp = nn.Sequential(
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

    def forward(self, t):
        emb = self.pos_emb(t)
        return self.mlp(emb)


class Conv1dBlock(nn.Module):
    """
    Single convolutional block

    Conv1d --> GroupNorm --> Mish

    #Todo:
        - Investigate GroupNorm alternatives.
        - Investigate Mish alternatives # Paper recommends Mish
        - Implement Unet-Model API Pipeline to Iterate different designs
        - Investigate padding as kernel_size // 2
        - Investigate number of groups as 8.

    #Done:
        - GroupNorm over BachNorm is still the best for CNN based Unet (LayerNorm,InstanceNorm are alternatives)
        - Mish over Relu is best non-monotonic, continuously differentiable, stability
        - padding = kernel_size // 2 : This is common practice to preserve spacial dim
        - group size 8 provides good trade-offs between channel groupinf and statistical robustness

    #Notes:
        - GroupNorm is generally preffered for CNN Unet
        - BatchSize Sensitivity and Compatibility with EMA training is the reason why BatchNorm is discouraged.
        - Similar alternatives to Mish includes Swish/SiLU but GELU is more for Transformers and ViT
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, n_groups=8, activation="Mish"
    ):
        super().__init__()

        # Todo: add different activation options
        # activations = {"Mish":nn.Mish(), "Swish":nn.Swish(), "SiLU":nn.SiLU()}
        self.conv1dblock = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),  # activations[activation]
        )

    def forward(self, x):
        return self.conv1dblock(x)


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class FiLMConditioning(nn.Module):
    """
    cond_dim: time embedding + global feature embedding # 250 + 514
    out_channel: block output channels
    """

    def __init__(self, cond_dim, out_channels):
        super().__init__()

        cond_channels = out_channels * 2
        self.film_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(
                cond_dim, cond_channels
            ),  # project cond_dim to out_channels * 2 # torch.Size([B, C])
            nn.Unflatten(-1, (-1, 1)),  # (B, C, 1)
        )

    def forward(self, cond):
        return self.film_encoder(cond)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()

        # group conv1d block operations
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(
                    in_channels, out_channels, kernel_size, n_groups=n_groups
                ),  # in, out
                Conv1dBlock(
                    out_channels, out_channels, kernel_size, n_groups=n_groups
                ),  # out, out
            ]
        )

        self.out_channels = out_channels

        # apply film conditioning to predict per-channel scale and bias
        self.film_embed = FiLMConditioning(cond_dim, out_channels)

        # make sure dimensions are compitible
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        """
        x (action_noise) : (batch_size, in_channels, horizon) -> (B, C, T)
        cond : (batch_size, cond_dim)

        #Notes:
        C : action dimensions
        cond: time_cond + obs_cond (256 + 514) -> (B, 770)

        returns:
        out : (batch_size, in_channels, horizon)
        """
        out = self.blocks[0](x)

        film_embed = self.film_embed(cond)
        film_embed = film_embed.reshape(film_embed.shape[0], 2, self.out_channels, 1)
        scale, bias = film_embed[:, 0, ...], film_embed[:, 1, ...]
        out = scale * out + bias
        #print(out.shape)

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class UnetEncoders(nn.Module):
    def __init__(self,
            in_out,
            cond_dim,
            kernel_size=5,
            n_groups=8):
        super().__init__()

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out): #(2,256),(256,512),(512,1024)
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        self.encoders = down_modules

    def __iter__(self):
        return iter(self.encoders)


class UnetLatent(nn.Module):
    def __init__(self,
            all_dims,
            cond_dim,
            kernel_size=5,
            n_groups=8):
        super().__init__()

        mid_dim = all_dims[-1]
        mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])
        self.latent = mid_modules

    def __iter__(self):
        return iter(self.latent)


class UnetUpSample(nn.Module):
    def __init__(self,
            in_out,
            cond_dim,
            kernel_size=5,
            n_groups=8):
        super().__init__()

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])): #(512,1024),(256,512)
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        self.decoders = up_modules

    def __iter__(self):
        return iter(self.decoders)


class ConditionalUnet1D(nn.Module):
    def __init__(self,
                input_dim,
                global_cond_dim,
                diffusion_step_embed_dim=256,
                down_dims=[256, 512, 1024],
                kernel_size=5,
                n_groups=8):

        """
            input_dim: Dim of noisy random actions for diffusion model noise prediction.
            global_cond_dim: Dim of encoded observation for FilM conditioning
            in addition to the diffusion step embedding. This is usually obs_horizon * obs_dim.
        """
        super().__init__()
        all_dims = [input_dim] + list(down_dims) #[input_dim, down_dims]
        start_dim = down_dims[0] #input_dim

        dsed = diffusion_step_embed_dim
        cond_dim = dsed + global_cond_dim #interger value

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        self.diffusion_step_encoder = SinusoidalEncoder(dsed)
        self.down_modules = UnetEncoders(in_out, cond_dim)
        self.mid_modules = UnetLatent(all_dims, cond_dim)
        self.up_modules = UnetUpSample(in_out, cond_dim)
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        print("number of parameters {:e}".format(
            sum(p.numel() for p in self.parameters())
        ))

    def forward(self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            global_cond=None):

        sample = sample.moveaxis(-1,-2) #from (B,T,C)->(B,C,T)

        timesteps = timestep
        if not torch.is_tensor(timesteps): # This is essential for inference from K:int -> torch
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps) #torch.Size([1, 256])

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1) # concat time embed with obs embed #torch.Size([1, 1284])

        x = sample
        h = []
        # downsample
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        # middle
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # upsample
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        #final
        x = self.final_conv(x)
        x = x.moveaxis(-1, -2) # return axis from (B,C,T)->(B,T,C)
        return x
