from tensorboard import summary
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision

from model.modules.dynamic_conv import DynamicConv
from model.modules.modules import SpatialAttention, activation
import config


class DyConvSR(nn.Module):
    def __init__(self,
                 num_channels=3,
                 num_feats=32,
                 num_blocks=1,
                 upscale=2,
                 act='gelu',
                 use_hfb=True,
                 spatial_attention=True,
                 ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.num_feats = num_feats
        self.num_blocks = num_blocks
        self.upscale = upscale
        self.use_hfb = use_hfb
        self.spatial_attention = spatial_attention
        
        self.activation = activation(act_type=act)
        self.gaussian = torchvision.transforms.GaussianBlur(kernel_size=5, sigma=1)
        
        self.pixelUnShuffle = nn.PixelUnshuffle(upscale)
        self.pixelShuffle = nn.PixelShuffle(upscale)
        
        ## ====== MAIN BRANCH ====== ##
        # head
        self.head = nn.Sequential(
            nn.Conv2d(num_channels * (upscale**2), num_feats, kernel_size=3, stride=1, padding=1)
        )
        # body
        body = []
        for i in range(num_blocks):
            body.append(DynamicConv(num_feats, num_feats, kernel_size=3, stride=1, padding=1, K=4, temperature=30, ratio=4))
            body.append(self.activation)
        self.body = nn.Sequential(*body)
        # tail
        self.tail = nn.Sequential(
            nn.Conv2d(num_feats, 48, kernel_size=3, stride=1, padding=1)
        )
        
        ## ====== HFB BRANCH ====== ##
        if use_hfb:
            hsb = []
            hsb.append(nn.Conv2d(num_channels * (upscale**2), 48, kernel_size=3, stride=1, padding=1))
            hsb.append(self.activation)
            self.hsb = nn.Sequential(*hsb)
            
        ## ====== SPATIAL ATTENTION ====== ##
        if spatial_attention:
            self.spatial_attention = SpatialAttention(kernel_size=7)
            
    def forward(self, x):
        # hfb branch
        hf = x - self.gaussian(x)
        hf_unsh = self.pixelUnShuffle(hf)
    
        # main branch
        out1 = self.pixelUnShuffle(x)
        out2 = self.head(out1)
        out3 = self.body(out2)
        out4 = self.tail(out3)
        
        if self.use_hfb:
            hf_deep_feats = self.hsb(hf_unsh)
            out4 = torch.add(out4, hf_deep_feats)
            
        if self.spatial_attention:
            out4 = out4 * self.spatial_attention(out4)

        out5 = self.pixelShuffle(out4)

        repeat = torch.cat((x, x, x, x), dim=1)
        out6 = torch.add(out5, repeat)
        out7 = self.pixelShuffle(out6)

        return out7


def dyconvsr(config):
    model = DyConvSR(
        num_channels=3,
        num_feats=config.num_feats,
        num_blocks=config.num_blocks,
        upscale=config.scale,
        act=config.act_type,
        use_hfb=config.use_hfb,
        spatial_attention=config.spatial_attention,
    )
    
    return model


if __name__ == "__main__":
    model = dyconvsr(config)
    
    summary(model, (3, 128, 128), device="cpu")