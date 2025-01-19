'''For Data Augmentation Component'''
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DataAugmentation:
    def __init__(self, dataset, num_augmented=4, batch_size=128, device='cuda'):
        if not isinstance(dataset, torch.Tensor):
            raise TypeError("Dataset must be a PyTorch tensor")

        if dataset.ndim != 4:
            raise ValueError("Dataset must have 4 dimensions (num_samples, channels, height, width)")

        self.dataset = dataset
        self.num_augmented = num_augmented
        self.batch_size = batch_size
        self.device = device

        # Define the augmentation transforms using albumentations
        self.augmentation_transforms = A.Compose([
            A.HorizontalFlip(),  # Horizontal flip
            A.Rotate(limit=(-30, 30)),  # Rotate between -30 and +30 degrees
            A.RandomResizedCrop(height=dataset.shape[2], width=dataset.shape[3], scale=(0.8, 1.0)),
            A.ElasticTransform(alpha=36, sigma=6),  # Elastic distortion
            ToTensorV2(),  # Convert numpy array to tensor
        ])

    def augment_batch(self, batch):
        """Applies augmentations to a single batch."""
        batch_augmented = []

        for i in range(len(batch)):
            data_sample = batch[i].numpy().transpose(1, 2, 0)  # Convert to HWC format

            for _ in range(self.num_augmented):
                augmented_sample = self.augmentation_transforms(image=data_sample)['image']
                batch_augmented.append(torch.tensor(augmented_sample, dtype=torch.float32))

        return batch_augmented

    def augment_dataset(self):
        """Applies augmentations to the entire dataset in batches."""
        augmented_data = []

        for batch_start in range(0, len(self.dataset), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(self.dataset))
            batch = self.dataset[batch_start:batch_end].cpu()  # Move batch to CPU for augmentation

            # Augment the batch
            batch_augmented = self.augment_batch(batch)

            # Concatenate original and augmented data
            original_batch = batch.to(self.device)
            augmented_batch = torch.stack(batch_augmented).to(self.device)
            batch_combined = torch.cat((original_batch, augmented_batch), dim=0)
            augmented_data.append(batch_combined)

            # Free memory
            del batch
            torch.cuda.empty_cache()

        if len(augmented_data) == 0:
            print("No augmented data was generated.")
            return torch.empty(0)

        # Concatenate all augmented data
        return torch.cat(augmented_data, dim=0)

