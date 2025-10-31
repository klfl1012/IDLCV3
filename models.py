import torch
import torch.nn as nn
from typing import List, Type, Optional


class ConvBlock(nn.Module):
    """Reusable conv block with residual connection"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Type[nn.Module],
        norm_type: str = 'batch',
        norm_groups: int = 8,
        dropout: Optional[float] = None,
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

        layers: List[nn.Module] = [
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


class EncoderBlock(nn.Module):
    """Single encoder stage: ConvBlock + Downsample"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Type[nn.Module],
        norm_type: str = 'batch',
        norm_groups: int = 8,
        dropout: Optional[float] = None,
        residual: bool = True,
    ):
        super().__init__()
        self.conv = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=activation,
            norm_type=norm_type,
            norm_groups=norm_groups,
            dropout=dropout,
            residual=residual,
        )
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (skip_connection, downsampled_output)"""
        skip = self.conv(x)
        down = self.pool(skip)
        return skip, down


class DecoderBlock(nn.Module):
    """Single decoder stage: Upsample + Concat + ConvBlock"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Type[nn.Module],
        norm_type: str = 'batch',
        norm_groups: int = 8,
        dropout: Optional[float] = None,
        residual: bool = True,
        use_skip_conn: bool = True,
    ):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )
        self.conv = ConvBlock(
            in_channels=out_channels * 2 if use_skip_conn else out_channels, # If using skip conn, input channels are doubled
            out_channels=out_channels,
            activation=activation,
            norm_type=norm_type,
            norm_groups=norm_groups,
            dropout=dropout,
            residual=residual,
        )
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if self.use_skip_conn and skip is not None:
            x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    """Complete encoder path with multiple stages"""
    
    def __init__(
        self,
        in_channels: int,
        features: List[int],
        activation: Type[nn.Module],
        norm_type: str,
        norm_groups: int,
        dropout: Optional[float],
        residual: bool = True,
    ):
        super().__init__()
        self.stages = nn.ModuleDict()
        self.num_stages = len(features)
        
        channels = in_channels
        for i, feat in enumerate(features):
            self.stages[f'stage_{i}'] = EncoderBlock(
                in_channels=channels,
                out_channels=feat,
                activation=activation,
                norm_type=norm_type,
                norm_groups=norm_groups,
                dropout=dropout if i > 0 else None,  # No dropout on first layer
                residual=residual,
            )
            channels = feat
    
    def forward(self, x: torch.Tensor) -> tuple[List[torch.Tensor], torch.Tensor]:
        """Returns (skip_connections, bottleneck_input)"""
        skips = []
        for i in range(self.num_stages):
            skip, x = self.stages[f'stage_{i}'](x)
            skips.append(skip)
        return skips, x


class Decoder(nn.Module):
    """Complete decoder path with multiple stages"""
    
    def __init__(
        self,
        features: List[int],
        bottleneck_channels: int,
        activation: Type[nn.Module],
        norm_type: str,
        norm_groups: int,
        dropout: Optional[float],
        residual: bool = True,
        use_skip_conn: bool = True,
    ):
        super().__init__()
        self.stages = nn.ModuleDict()
        self.num_stages = len(features)
        
        channels = bottleneck_channels
        for i, feat in enumerate(reversed(features)):
            self.stages[f'stage_{i}'] = DecoderBlock(
                in_channels=channels,
                out_channels=feat,
                activation=activation,
                norm_type=norm_type,
                norm_groups=norm_groups,
                dropout=dropout if i < len(features) - 1 else None,  # No dropout on last layer
                residual=residual,
                use_skip_conn=use_skip_conn,
            )
            channels = feat
    
    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        for i in range(self.num_stages):
            x = self.stages[f'stage_{i}'](x, skips[-(i+1)])
        return x


class UNet(nn.Module):
    """
    U-Net with separated Encoder and Decoder classes. 
    Can be used as a Encoder-Decoder model without skip connections.

    Args:    
        in_channels: Number of input channels
        out_channels: Number of output channels
        features: List of feature map sizes for each encoder stage
        activation: Activation function class
        norm_type: Type of normalization ('batch', 'group', 'instance')
        norm_groups: Number of groups for GroupNorm (if used)
        dropout: Dropout probability
        residual: Whether to use residual connections in ConvBlocks
        use_skip_conn: Whether to use skip connections between encoder and decoder

    Returns:
        nn.Module: U-Net model or Encoder-Decoder model
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        features: Optional[List[int]] = None,
        activation: Type[nn.Module] = nn.SiLU,
        norm_type: str = 'batch',
        norm_groups: int = 8,
        dropout: Optional[float] = 0.1,
        residual: bool = True,
        use_skip_conn: bool = True,
    ):
        super().__init__()
        features = features or [64, 128, 256, 512]
        
        # Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            features=features,
            activation=activation,
            norm_type=norm_type,
            norm_groups=norm_groups,
            dropout=dropout,
            residual=residual,
        )
        
        # Bottleneck
        bottleneck_channels = features[-1] * 2
        self.bottleneck = ConvBlock(
            in_channels=features[-1],
            out_channels=bottleneck_channels,
            activation=activation,
            norm_type=norm_type,
            norm_groups=norm_groups,
            dropout=dropout,
            residual=residual,
        )
        
        # Decoder
        self.decoder = Decoder(
            features=features,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            norm_type=norm_type,
            norm_groups=norm_groups,
            dropout=dropout,
            use_skip_conn=use_skip_conn,
            residual=residual,
        )
        
        # Head
        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        skips, x = self.encoder(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        x = self.decoder(x, skips)
        
        # Output head
        return self.head(x)




def create_compiled_unet(
    in_channels: int,
    out_channels: int,
    features: Optional[List[int]] = None,
    activation: Type[nn.Module] = nn.SiLU,
    norm_type: str = 'batch',
    norm_groups: int = 8,
    dropout: Optional[float] = 0.1,
    residual: bool = True,
    use_skip_conn: bool = True,
    enable_compile: bool = True,
    compile_mode: str = 'default',
):

    model = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        features=features,
        activation=activation,
        norm_type=norm_type,
        norm_groups=norm_groups,
        dropout=dropout,
        residual=residual,
        use_skip_conn=use_skip_conn,
    )
    
    # Apply torch.compile if available and requested
    should_compile = (
        enable_compile and 
        torch.__version__ >= '2.0.0' and
        torch.cuda.is_available()
    )
    
    if should_compile:
        model = torch.compile(model, mode=compile_mode) 
        print(f'torch.compile enabled with mode="{compile_mode}"')
    else:
        if enable_compile:
            print('torch.compile requires PyTorch 2.0+ and CUDA. Proceeding without compilation.')
    
    # Enable cuDNN autotuner for better performance
    torch.backends.cudnn.benchmark = True
    
    return model