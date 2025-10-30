import torch.nn as nn
import torch

# TODO: Dropout only during training,
# TODO: mixed precision support,
# TODO: torch.compile and autocast,
# TODO: batch norm instead of group norm,
# TODO: kernel fusion

class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: type[nn.Module],
        norm_type: str = 'batch',
        norm_groups: int = 8,
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

        if norm_type == 'batch':
            norm_layer1 = nn.BatchNorm2d(out_channels)
            norm_layer2 = nn.BatchNorm2d(out_channels)
        elif norm_type == 'group':
            norm_layer1 = nn.GroupNorm(norm_groups, out_channels)
            norm_layer2 = nn.GroupNorm(norm_groups, out_channels)
        elif norm_type == 'instance':
            norm_layer1 = nn.InstanceNorm2d(out_channels, affine=False)
            norm_layer2 = nn.InstanceNorm2d(out_channels, affine=False)

        else:
            raise ValueError(f'Unknown norm_type: {norm_type}')

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer1,
            activation(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer2
        ]

        if dropout:
            self.dropout = nn.Dropout2d(p=dropout)

        else:
            self.dropout = None

        self.block = nn.Sequential(*layers)
        self.final_act = activation(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.dropout is not None and self.training:
            out = self.dropout(out)

        if self.residual:
            out = out + self.skip(x)

        return self.final_act(out)
    

class UNet(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        features: list[int] | None = None,
        activation: type[nn.Module] = nn.SiLU,
        norm_type: str = 'batch',
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
                    in_channels=channels,
                    out_channels=feat,
                    activation=activation,
                    norm_type=norm_type,
                    norm_groups=norm_groups,
                    dropout=dropout if channels != in_channels else None,
                )
            )
            self.downsample.append(nn.MaxPool2d(2, 2))
            channels = feat

        bottleneck_channels = features[-1] * 2
        self.bottleneck = ConvBlock(
            in_channels=features[-1],
            out_channels=bottleneck_channels,
            activation=activation,
            norm_type=norm_type,
            norm_groups=norm_groups,
            dropout=dropout,
        )

        for feat in reversed(features):
            # self.upsample.append(
            #     nn.ConvTranspose2d(
            #         bottleneck_channels, feat, kernel_size=2, stride=2
            #     )
            # )

            # self.upsample.append(
            #     nn.Sequential(
            #         nn.Upsample(scale_factor=2, mode='nearest'), 
            #         nn.Conv2d(bottleneck_channels, feat, kernel_size=3, padding=1)
            #     )
            # )

            self.upsample.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        bottleneck_channels,
                        feat,
                        kernel_size=2,
                        stride=2
                    )
                )
            )

            self.up_blocks.append(
                ConvBlock(
                    in_channels=feat * 2,
                    out_channels=feat,
                    activation=activation,
                    norm_type=norm_type,
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

        # Encoder
        for down, pool in zip(self.down_blocks, self.downsample):
            x = down(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for upsample, block, skip in zip(self.upsample, self.up_blocks, reversed(skips)):
            x = upsample(x)
            # if x.shape[-2:] != skip.shape[-2:]:
            #     x = nn.functional.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = block(x)
        
        return self.head(x)
    

def create_compiled_unet(
    in_channels: int, 
    out_channels: int,
    features: list[int] | None = None,
    activation: type[nn.Module] = nn.SiLU,
    norm_type: str = 'batch',
    norm_groups: int = 8,
    dropout: float | None = 0.1,
    mode: str = 'default', # 'default' | 'max-autotune' | 'reduce-overhead'
):
    
    model = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        features=features,
        activation=activation,
        norm_type=norm_type,
        norm_groups=norm_groups,
        dropout=dropout,
    )

    if torch.__version__ >= '2.0.0':
        model = torch.compile(model, mode=mode)
        print(f'torch.compile enabled with mode="{mode}"')

    else:
        print('torch.compile requires PyTorch 2.0.0 or higher. Proceeding without compilation.')

    torch.backends.cudnn.benchmark = True

    return model



if __name__ == '__main__':

    base_model = UNet(in_channels=3, out_channels=1, norm_type='batch')
    
    from torchinfo import summary
    print("\n=== Unkompiliertes Modell (für Summary) ===")
    summary(base_model, input_size=(1, 3, 256, 256))

    print("\n=== Performance-Test ===")
    import time
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    x = torch.randn(4, 3, 256, 256).to(device)
    
    model_normal = UNet(3, 1, norm_type='batch').to(device).eval()
    model_compiled = create_compiled_unet(3, 1, norm_type='batch', mode='max-autotune')
    model_compiled = model_compiled.to(device).eval()
    
    with torch.no_grad():
        for _ in range(10):  # Warmup
            _ = model_normal(x)
            _ = model_compiled(x)
    
    n_runs = 100
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model_normal(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    time_normal = time.time() - start
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model_compiled(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    time_compiled = time.time() - start
    
    print(f"\nNormal model:    {time_normal:.4f}s ({n_runs} runs)")
    print(f"Compiled model:  {time_compiled:.4f}s ({n_runs} runs)")
    print(f"Speedup:         {time_normal/time_compiled:.2f}x")

    