'''Importing Libraries'''
import numpy as np
import torch
import math
import math
from huggingface_hub import HfApi
from huggingface_hub import login
from diffusers import DDPMScheduler
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline
from accelerate import notebook_launcher


from Parameters.parameters import TrainingConfig
from Parameters.parameters import UNetConfig
from Data_Augmentation.data_augmentation import DataAugmentation
from Data_Transformation.data_transformation import NormalizeTransform
from Data_Transformation.data_transformation import DataResizer
from U_Net.u_net import UNetModel
from Trainer.trainer import Trainer
from Utils.utils import ImageEvaluator


# Logging into Huggingface
token = os.getenv("HUGGINGFACE_TOKEN")
login(token=token)

#Setting up model pusher
repo_id = "2DseisvelGenerator"
user = "kankur0007"

api = HfApi()
api.create_repo(repo_id=repo_id, token=token, exist_ok=True)

"""-----------------------------Data_Ingestion & Transformation Configuration------------------------------------------"""
# Loading Data
Data = np.load('Velocity_Models/Seismic_Velocity_Models.npy')
# Convert to PyTorch tensor
dataset1 = torch.from_numpy(Data)
# Applying the data augmentation
dataset2 = DataAugmentation(dataset1)
"""Initialize the transformation class"""
# Compute mean and std from the dataset
mean = dataset1.mean()
std = dataset1.std()
transform = NormalizeTransform(mean, std)
# Apply normalization
dataset3 = transform.apply_transforms(dataset1, method="normalize")
# Initialize the resizer
resizer = DataResizer(size=(64, 64))
# Resize the dataset
dataset4 = resizer.resize(dataset2)
"""---------------------------------Setting Up Training Configuration---------------------------------------------------------"""
config= TrainingConfig()
train_dataloader = torch.utils.data.DataLoader(dataset4, batch_size=config.train_batch_size, shuffle=True)
# Defining the model 
unetconfig = UNetConfig()
model = UNetModel(config=unetconfig)
# Defining the loss function
loss = F.mse_loss(noise_pred, noise)
# Defining the Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,weight_decay=1e-4)
# Defing warmup
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer, # Specifies the optimizer for which the learning rate schedule will be applied.
    num_warmup_steps=config.lr_warmup_steps, # Sets the number of warmup steps, during which the learning rate increases linearly from 0 to the initial learning rate.
    num_training_steps=(len(train_dataloader) * config.num_epochs), # This determines the total number of steps over which the cosine annealing schedule will be applied.
)
# Calling Evaluation Function
evaluatte = ImageEvaluator(output_dir=config.output_dir, seed=config.seed)

"""----------------------------------------------Now training the model----------------------------------------------------------"""
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
trainer = Trainer(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
notebook_launcher(train_loop, args, num_processes=1) # Lets train

