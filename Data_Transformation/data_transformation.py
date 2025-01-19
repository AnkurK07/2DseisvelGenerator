import torch
from torchvision import transforms

class NormalizeTransform:
    def __init__(self, mean, std):
        """
        Initialize the NormalizeTransform class.

        Args:
        - mean (torch.Tensor or list): Mean used for normalization (e.g., [0.5]).
        - std (torch.Tensor or list): Std used for normalization (e.g., [0.2]).
        """
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.normalize = transforms.Normalize(mean=self.mean.tolist(), std=self.std.tolist())

    def normalize_image(self, img):
        """
        Apply normalization to a single image.

        Args:
        - img (torch.Tensor): Image tensor with shape (C, H, W).

        Returns:
        - torch.Tensor: Normalized image tensor.
        """
        return self.normalize(img)

    def unnormalize_image(self, img):
        """
        Revert normalization for a single image.

        Args:
        - img (torch.Tensor): Normalized image tensor with shape (C, H, W).

        Returns:
        - torch.Tensor: Unnormalized image tensor.
        """
        return img * self.std[0] + self.mean[0]

    def apply_transforms(self, dataset, method="normalize"):
        """
        Apply normalization or unnormalization to the entire dataset.

        Args:
        - dataset (torch.Tensor): Dataset tensor with shape (N, C, H, W).
        - method (str): 'normalize' or 'unnormalize'.

        Returns:
        - torch.Tensor: Transformed dataset tensor.
        """
        transformed_dataset = []
        if method == "normalize":
            for img in dataset:
                transformed_dataset.append(self.normalize_image(img))
        elif method == "unnormalize":
            for img in dataset:
                transformed_dataset.append(self.unnormalize_image(img))
        else:
            raise ValueError("Invalid method. Choose 'normalize' or 'unnormalize'.")

        return torch.stack(transformed_dataset)


