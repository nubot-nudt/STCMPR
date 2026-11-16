"""Dual-branch descriptor network; checkpoint parameter names are preserved.

Unused legacy modules are retained for strict loading of existing checkpoints.
Pretrained weights are explicit; construction never downloads a ResNet checkpoint.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models import netvlad
from .mamba_vision import MambaVision


class Backbone(nn.Module):
    def __init__(self, nmf=False, mlp=False, pretrained=None):
        super(Backbone, self).__init__()

        self.nmf = nmf
        self.K = 16
        self.max_iter = 50
        print("nmf: ", self.nmf)
        print("K: ", self.K)

        self.mlp = mlp
        print("mlp: ", self.mlp)

        encoder = models.resnet34(pretrained=False)  # resnet34 capture only features and remove last relu and maxpool
        layers = list(encoder.children())[:-3]
        self.encoder = nn.Sequential(*layers)
        self.relu = nn.ReLU(inplace=True)
        self.model_path = pretrained
        self.mamba_vision1 = MambaVision(depths=[3, 3, 10, 5],
                        num_heads=[2, 4, 8, 16],
                        window_size=[8, 8, 14, 7],
                        dim=128,
                        in_dim=64,
                        mlp_ratio=4,
                        resolution=224,
                        drop_path_rate=0.3,
                        layer_scale=1e-5,
                        layer_scale_conv=None)
        if pretrained:
            self.mamba_vision1._load_state_dict(pretrained)

        self.mamba_vision2 = MambaVision(depths=[3, 3, 10, 5],
                        num_heads=[2, 4, 8, 16],
                        window_size=[8, 8, 14, 7],
                        dim=128,
                        in_dim=64,
                        mlp_ratio=4,
                        resolution=224,
                        drop_path_rate=0.3,
                        layer_scale=1e-5,
                        layer_scale_conv=None)
        if pretrained:
            self.mamba_vision2._load_state_dict(pretrained)

        self.mlp_c = nn.Sequential(nn.Conv2d(256, 128, kernel_size=(1, 1)),
                                   nn.ReLU(),
                                   nn.Conv2d(128, self.K, kernel_size=(1, 1)),
                                   nn.ReLU())
        self.mamba_mlp1 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=(1, 1)),
                                        nn.ReLU(),
                                        nn.Conv2d(512, 256, kernel_size=(1, 1)),
                                        nn.ReLU()
                                        )
        self.mamba_mlp2 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=(1, 1)),
                                        nn.ReLU(),
                                        nn.Conv2d(512, 256, kernel_size=(1, 1)),
                                        nn.ReLU()
                                        )
        self.net_vlad_cnn1 = netvlad.NetVLAD(num_clusters=64, dim=256, vladv2=True)
        self.net_vlad_cnn2 = netvlad.NetVLAD(num_clusters=64, dim=256, vladv2=True)
        self.net_vlad_nmf = netvlad.NetVLAD(num_clusters=64, dim=self.K, vladv2=True)
        self.net_vlad_mlp = netvlad.NetVLAD(num_clusters=64, dim=self.K, vladv2=True)

    def forward(self, x):
        tmp = int(x.shape[0] / 2)
        x_img = x[:tmp, :, :, :]
        x_sam = x[tmp:, :, :, :]

        # depth
        x_img = self.mamba_vision1(x_img)
        x_img = F.normalize(x_img, p=2, dim=1)
        x_img = self.mamba_mlp1(x_img)
        x_img = F.normalize(x_img, p=2, dim=1)
        x_mid = x_img
        x_img = self.net_vlad_cnn1(x_img)
        x_img = F.normalize(x_img, p=2, dim=1)

        # MobileSAM
        x_sam = self.mamba_vision2(x_sam)
        x_sam = F.normalize(x_sam, p=2, dim=1)
        x_sam = self.mamba_mlp2(x_sam)
        x_sam = F.normalize(x_sam, p=2, dim=1)
        # x_mid = x_sam
        x_sam = self.net_vlad_cnn2(x_sam)
        x_sam = F.normalize(x_sam, p=2, dim=1)

        out = torch.cat((x_img, x_sam), 1)

        return x_mid, out


class TripletLossSimple(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLossSimple, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = torch.sqrt((anchor - positive).pow(2).sum(1))
        neg_dist = torch.sqrt((anchor - negative).pow(2).sum(1))
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

