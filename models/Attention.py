import torch
import torch.nn as nn
import torch.nn.functional as F
from .AdaIN import *
import fvcore.nn.weight_init as weight_init
    
class CrossAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(CrossAttentionBlock, self).__init__()
        
        self.g = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1, padding=0)

        self.W = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1, stride=1, padding=0)
        for layer in [self.g, self.theta, self.phi, self.W]:
            weight_init.c2_xavier_fill(layer)
        
        self.adain = AdaIN_block()
        
    def forward(self, x):
        batch_size, C, height, width = x[0].size()

        g_x = self.g(x[0]).view(batch_size, -1, height * width).permute(0, 2, 1)

        theta_x = self.theta(x[1]).view(batch_size, -1, height * width)
        phi_x = self.phi(x[1]).view(batch_size, -1, height * width).permute(0, 2, 1)
        
        f = torch.matmul(phi_x, theta_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x).view(batch_size, C // 8, height, width)
        W_y = self.W(y)
        z = self.adain(x[0], W_y)
        return z
    
class EfficientCrossAttention(nn.Module):

    def __init__(self, in_channels_x, in_channels_y, key_channels, head_count, value_channels):
        super().__init__()
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels_y, key_channels, kernel_size=1, stride=1, padding=0)
        self.queries = nn.Conv2d(in_channels_x, key_channels, kernel_size=1, stride=1, padding=0)
        self.values = nn.Conv2d(in_channels_y, value_channels, kernel_size=1, stride=1, padding=0)
        self.reprojection = nn.Conv2d(value_channels, in_channels_x, kernel_size=1, stride=1, padding=0)

        self.adain = AdaIN_block()

    def forward(self, z):
        x = z[0]
        y = z[1]
        n, _, h, w = x.size()
        queries = self.queries(x).reshape(n, self.key_channels, h * w)
        keys = self.keys(y).reshape((n, self.key_channels, h * w))
        values = self.values(y).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[
                            :,
                            i * head_key_channels: (i + 1) * head_key_channels,
                            :
                            ], dim=2)
            query = F.softmax(queries[
                              :,
                              i * head_key_channels: (i + 1) * head_key_channels,
                              :
                              ], dim=1)
            value = values[
                    :,
                    i * head_value_channels: (i + 1) * head_value_channels,
                    :
                    ]
            context = key @ value.transpose(1, 2)
            attended_value = (
                    context.transpose(1, 2) @ query
            ).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)
        out = self.adain(x, attention)

        return out
    # query from x, key and value from y. After we get the attention map, we apply the AdaIN to the input whose give the value information?
    # If we use the AdaIN to the input whose give the query information, it might be extracting more fine-grained features but less broader contextual information?