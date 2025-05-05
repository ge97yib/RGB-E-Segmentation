import torch.nn as nn

class AdaIN_block(nn.Module):
    def __init__(self):
        super(AdaIN_block, self).__init__()

    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def forward(self, rgb, evt):
        # print(rgb.size()) torch.Size([8, 256/512/1024/2048, 87, 87])

        assert (rgb.size()[:2] == evt.size()[:2])
        size = rgb.size()
        style_mean, style_std = self.calc_mean_std(evt)
        content_mean, content_std = self.calc_mean_std(rgb)
        normalized_feat = (rgb - content_mean.expand(
            size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)