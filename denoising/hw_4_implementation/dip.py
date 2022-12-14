import numpy as np
import torch
import torch.nn as nn


class EncDec(nn.Module):

    def __init__(self):
        super(EncDec, self).__init__()

        # Encoder part of network

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
        )

        # Decoder part of network

        self.layer4 = nn.Sequential(
            nn.UpsamplingBilinear2d((130, 90)),
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=1,
            ),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
        )

        self.layer5 = nn.Sequential(
            nn.UpsamplingBilinear2d((275, 175)),
            nn.Conv2d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=1,
            ),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
        )

        self.layer6 = nn.Sequential(
            nn.UpsamplingBilinear2d((382, 254)),
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=1,
                kernel_size=3,
                stride=1,
            ),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
        )

        # Output Size Should Be: (1, 1, 384, 256)

    def forward(self, x):
        l1_out = self.layer1(x)
        l2_out = self.layer2(l1_out)
        l3_out = self.layer3(l2_out)
        l4_out = self.layer4(l3_out)
        l5_out = self.layer5(l4_out)
        l6_out = self.layer6(l5_out)
        return l6_out
