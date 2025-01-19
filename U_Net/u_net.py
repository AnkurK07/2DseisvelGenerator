
from diffusers import UNet2DModel
class UNetModel:
    """
    U-Net model wrapper with initialization only.
    """
    def __init__(self, config:None):
        """
        Initialize the U-Net model with the given configuration.

        Args:
        - config (UNetConfig): Configuration object containing U-Net parameters.
        """
        self.config = config
        self.model = UNet2DModel(
            sample_size=config.image_size,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            layers_per_block=config.layers_per_block,
            block_out_channels=config.block_out_channels,
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )