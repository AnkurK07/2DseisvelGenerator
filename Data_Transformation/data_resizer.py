import torch
import torchvision.transforms.functional as F

class DataResizer:
    def __init__(self, size):
        """
        Initialize the DataResizer class.

        Args:
        - size (tuple): Target size for resizing (height, width).
        """
        self.size = size

    def resize(self, tensor):
        """
        Resize a single image or a batch of images.

        Args:
        - tensor (torch.Tensor): Input tensor with shape (N, C, H, W) for batches 
          or (C, H, W) for a single image.

        Returns:
        - torch.Tensor: Resized tensor with updated spatial dimensions.
        """
        if tensor.ndim == 4:
            # Resize each image in the batch
            resized_images = [F.resize(img, self.size) for img in tensor]
            return torch.stack(resized_images)
        elif tensor.ndim == 3:
            # Resize single image
            return F.resize(tensor, self.size)
        else:
            raise ValueError("Unsupported tensor dimensions. Expected 3 or 4 dimensions.")
