import torch
import torch.nn as nn
import torchvision
from transformers import AutoModel, AutoConfig

# ----------------------------
# Existing resnet logic
# ----------------------------
def get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    # load r3m weights
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)

    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    return resnet

def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m
    r3m.device = 'cpu'
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to('cpu')
    return resnet_model

# ----------------------------
# Hugging Face wrapper
# ----------------------------
class EncoderModel(nn.Module):
    def __init__(self, checkpoint='bert-base-uncased', pretrained=True):
        super().__init__()
        if pretrained:
            self.model = AutoModel.from_pretrained(
                checkpoint,
                device_map="cpu"
            )
        else:
            config = AutoConfig.from_pretrained(checkpoint)
            self.model = AutoModel.from_config(config)

    def forward(self, x):
        # return pooled embedding only
        return self.model(x).pooler_output


# ----------------------------
# Unified entry point
# ----------------------------
def get_rgb_model(name, weights=None, pretrained=True, **kwargs):
    """
    Entry point for encoders.

    - ResNets come from torchvision / r3m
    - Other models come from Hugging Face hub
    """
    if name.startswith("resnet"):
        return get_resnet(name, weights=weights, **kwargs)
    else:
        # treat `name` as a Hugging Face checkpoint
        return EncoderModel(checkpoint=name, pretrained=pretrained)