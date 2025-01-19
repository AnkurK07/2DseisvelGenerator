import os
import math
import torch
from PIL import Image

class ImageEvaluator:
    def __init__(self, output_dir: str, seed: int = 42):
        """
        Initialize the ImageEvaluator class.

        Args:
        - output_dir (str): Directory to save the generated images.
        - seed (int): Seed for reproducibility.
        """
        self.output_dir = output_dir
        self.seed = seed

    @staticmethod
    def make_grid(images, rows, cols):
        """
        Create a grid of images.

        Args:
        - images (list of PIL.Image): List of images to arrange in a grid.
        - rows (int): Number of rows in the grid.
        - cols (int): Number of columns in the grid.

        Returns:
        - PIL.Image: Image grid.
        """
        w, h = images[0].size
        grid = Image.new('RGB', size=(cols * w, rows * h))
        for i, image in enumerate(images):
            grid.paste(image, box=(i % cols * w, i // cols * h))
        return grid

    def evaluate(self, config, epoch, pipeline):
        """
        Evaluate the model by generating and saving a grid of images.

        Args:
        - config (object): Configuration object with `eval_batch_size` and `output_dir`.
        - epoch (int): Current epoch number.
        - pipeline (object): Diffusion pipeline for generating images.
        """
        # Generate images using the pipeline
        images = pipeline(
            batch_size=config.eval_batch_size,
            generator=torch.manual_seed(self.seed),
        ).images

        # Create an image grid
        image_grid = self.make_grid(images, rows=4, cols=4)

        # Save the image grid
        test_dir = os.path.join(config.output_dir, "samples")
        os.makedirs(test_dir, exist_ok=True)
        image_grid.save(f"{test_dir}/{epoch:04d}.png")
