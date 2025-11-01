import torch.nn as nn
import torch

class ConvBMAVBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float,
        activation: type[nn.Module],
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()

        self.body = nn.Sequential(  
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True),
        )

        if dropout:
            self.body.add_module('dropout', nn.Dropout(dropout))

        self.final_act = activation(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.body(x)
        return self.final_act(out)
    
class EncoderDecoderModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float | None = 0.1,
        features: list[int] | None = None,
        activation: type[nn.Module] = nn.ReLU,
        bottleneck_channel_multiplier: int = 1,
    ):
        super().__init__()
        features_maps = features or [64, 128, 256, 512]
        
        # 1. Encoder
        self.encoder = nn.Sequential()
        channels = in_channels
        for feature_map in features_maps:
            self.encoder.add_module('ConvBMAVBlock',                 
                ConvBMAVBlock(
                    channels,
                    feature_map,
                    dropout if channels != in_channels else None,
                    activation,
                    kernel_size,
                    stride,
                    padding,
                )
            )
            self.encoder.add_module('MaxPooling',nn.MaxPool2d(2, 2))

        # 2. Bottleneck
        bottleneck_channels = features_maps[-1] * bottleneck_channel_multiplier
        self.bottleneck_conv = nn.Conv2d(features_maps[-1], bottleneck_channels, kernel_size=kernel_size, padding=1)

        # 3. Decoder
        self.decoder = nn.Sequential()
        features_maps.append(bottleneck_channels)
        reversed_feature_map_order = list(reversed(features_maps))   
        for i in range(len(reversed_feature_map_order) - 1):
            self.decoder.add_module('Upsample',nn.Upsample(scale_factor=2, mode='nearest'))
            self.decoder.add_module('Conv2d',nn.Conv2d(bottleneck_channels, feature_map, kernel_size=kernel_size, padding=1))
    
            self.decoder.add_module('ConvBMAVBlock',
                ConvBMAVBlock(
                    reversed_feature_map_order[i],
                    reversed_feature_map_order[i + 1],
                    dropout=dropout if reversed_feature_map_order[i] != reversed_feature_map_order[-1] else None,
                    activation=activation,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )

        self.head = nn.Conv2d(reversed_feature_map_order[-1], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Encoder
        x = self.encoder(x)

        # 2. Bottleneck
        x = self.bottleneck_conv(x)

        # 3. Decoder
        x = self.decoder(x)
        return self.head(x)