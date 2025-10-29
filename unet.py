import torch.nn as nn
import torch



class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: type[nn.Module],
        norm_groups: int,
        dropout: float | None = None,
        residual: bool = True,
    ):
        super().__init__()
        self.residual = residual

        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, out_channels),
            activation(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, out_channels)
        ]

        if dropout:
            layers.append(nn.Dropout2d(dropout))

        self.block = nn.Sequential(*layers)
        self.final_act = activation(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.residual:
            out += self.skip(x)
        return self.final_act(out)
    

class UNet(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        features: list[int] | None = None,
        activation: type[nn.Module] = nn.SiLU,
        norm_groups: int = 8,
        dropout: float | None = 0.1,
    ):
        super().__init__()
        features = features or [64, 128, 256, 512]

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        self.upsample = nn.ModuleList()

        channels = in_channels
        for feat in features:
            self.down_blocks.append(
                ConvBlock(
                    channels,
                    feat,
                    activation,
                    norm_groups,
                    dropout if channels != in_channels else None,
                )
            )
            self.downsample.append(nn.MaxPool2d(2, 2))
            channels = feat

        bottleneck_channels = features[-1] * 2
        self.bottleneck = ConvBlock(
            features[-1],
            bottleneck_channels,
            activation=activation,
            norm_groups=norm_groups,
            dropout=dropout,
        )

        for feat in reversed(features):
            # self.upsample.append(
            #     nn.ConvTranspose2d(
            #         bottleneck_channels, feat, kernel_size=2, stride=2
            #     )
            # )

            self.upsample.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'), 
                    nn.Conv2d(bottleneck_channels, feat, kernel_size=3, padding=1)
                )
            )

            self.up_blocks.append(
                ConvBlock(
                    feat * 2,
                    feat,
                    activation=activation,
                    norm_groups=norm_groups,
                    dropout=dropout if feat != features[0] else None,
                )
            )
            bottleneck_channels = feat

        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []

        for down, pool in zip(self.down_blocks, self.downsample):
            x = down(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)
        for upsample, block, skip in zip(self.upsample, self.up_blocks, reversed(skips)):
            x = upsample(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = block(x)
        
        return self.head(x)
    
