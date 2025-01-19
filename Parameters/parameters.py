'''For Model Parameters'''
# Data Class For Whole Dataset
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 64  # Specifies the resolution of the generated images, which in this case is set to 64x64 pixels.
    train_batch_size = 128 # Defines the batch size for training, meaning that the model will process 1024 images per training step.
    eval_batch_size = 1024  # Specifies the batch size for evaluation, i.e., how many images will be sampled during the evaluation phase.
    num_epochs = 500  # Sets the number of epochs, meaning the model will go through the entire training dataset 800 times during training.
    gradient_accumulation_steps = 8 #  This controls gradient accumulation, which means accumulating gradients over multiple batches before performing an update. A value of 1 means
    # no accumulation (i.e., update every batch).
    learning_rate = 1e-4 # Sets the learning rate for the optimizer. This controls the step size at each iteration while moving toward a minimum of the loss function.
    lr_warmup_steps = 1000 # Indicates the number of steps to gradually increase the learning rate at the start of training (warmup).
    save_image_epochs = 50 # Specifies that the model will save generated images every 100 epochs during training.
    save_model_epochs = 50  # The model will be saved every 50 epochs during training.
    mixed_precision = 'fp16'  # Specifies the use of mixed precision for training. 'fp16' means the model will use 16-bit floating-point numbers for faster computation and reduced
    # memory usage. 'no' would indicate 32-bit precision (default).
    output_dir = '2DseisvelGenerator'  # Defines the directory where the model will be saved locally and potentially uploaded to the Hugging Face (HF) Hub.

    push_to_hub = True  # Indicates whether to automatically upload the model to the Hugging Face Hub once training is complete.
    hub_private_repo = False # If set to True, the model will be uploaded to a private repository on the Hugging Face Hub. False means it will be a public repository.
    overwrite_output_dir = True  #  Allows the model directory to be overwritten if you re-run the notebook. This is useful if you're iterating on the training process and want to
    # replace older versions.
    seed = 0 # This sets the random seed for reproducibility. Setting it to 0 ensures that the training results can be reproduced exactly.


'''U-net Parameters'''
@dataclass
class UNetConfig:
    """
    Configuration for the U-Net model.
    """
    image_size: int = 64  # Target image resolution
    in_channels: int = 1  # Number of input channels (1 for grayscale, 3 for RGB)
    out_channels: int = 1  # Number of output channels
    layers_per_block: int = 2  # ResNet layers per U-Net block
    block_out_channels: tuple = (64, 64, 128, 128, 256, 256)  # Channels per block