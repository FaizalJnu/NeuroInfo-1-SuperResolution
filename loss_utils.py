import torch
import torch.nn as nn
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Load VGG19 pretrained on ImageNet
        vgg = models.vgg19(pretrained=True).features.to(device)
        
        # We only need the first few layers (texture/edge detection)
        # Slicing up to layer 35 usually captures high-level structure + texture
        self.loss_network = nn.Sequential(*list(vgg.children())[:35]).eval()
        
        # Freeze parameters (we don't train VGG)
        for param in self.loss_network.parameters():
            param.requires_grad = False
            
    def forward(self, fake, real):
        # VGG expects 3 channels. If we have 1 channel, repeat it.
        if fake.shape[1] == 1:
            fake = fake.repeat(1, 3, 1, 1)
            real = real.repeat(1, 3, 1, 1)
            
        # Normalize for VGG (ImageNet mean/std approximation)
        # This simple scaling usually works well enough for MRI in 0-1 range
        fake = (fake - 0.485) / 0.229
        real = (real - 0.485) / 0.229
        
        # Get features
        feat_fake = self.loss_network(fake)
        feat_real = self.loss_network(real)
        
        # L1 loss between features
        return nn.functional.l1_loss(feat_fake, feat_real)