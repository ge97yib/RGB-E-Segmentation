import torch
import torch.nn as nn

from .Attention import *


class REFusion(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(REFusion, self).__init__()
    
       
        self.rgb_cross_attention = CrossAttentionBlock(in_planes)
        self.event_cross_attention = CrossAttentionBlock(in_planes)
        

        self.conv0_rgb = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0)
        self.conv0_evt = nn.Conv2d(out_planes, out_planes, kernel_size=1, stride=1, padding=0)

    
    def forward(self, rgb, evt):

            
        rgb0 = self.conv0_rgb(rgb)
        evt0 = self.conv0_evt(evt)
          
        mul = rgb0.mul(evt0)
        rgb1 = rgb0 + mul
        evt1 = evt0 + mul
            
     
        rgb_y = self.rgb_cross_attention([rgb1, evt1])
        event_y = self.event_cross_attention([evt1, rgb1])
          
        out = torch.cat([rgb_y, event_y], dim=1)
     
        return out
