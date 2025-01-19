'''Model Trainer loop '''
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, HfFolder, whoami
from accelerate import Accelerator
from tqdm.auto import tqdm

class Trainer:
    def __init__(self, config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
        self.config = config
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.lr_scheduler = lr_scheduler

        # Initialize Accelerator
        self.accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=os.path.join(config.output_dir, "logs")
        )

        # Hugging Face API and repository setup
        self.api = HfApi()
        self.token = HfFolder.get_token()
        self.repo_name = self._get_repo_name()

        if self.accelerator.is_main_process:
            self._prepare_repository()

        # Prepare the model, optimizer, and dataloaders with Accelerator
        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )

        self.global_step = 0
        self.epoch_losses = []

    def _get_repo_name(self):
        """Helper method to get the full repository name."""
        model_id = Path(self.config.output_dir).name
        if self.config.organization:
            return f"{self.config.organization}/{model_id}"
        username = whoami(self.token)["name"]
        return f"{username}/{model_id}"

    def _prepare_repository(self):
        """Prepare Hugging Face Hub repository."""
        if self.config.push_to_hub:
            self.api.create_repo(self.repo_name, exist_ok=True, token=self.token)
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.accelerator.init_trackers("train_example")

    def train(self):
        """Training loop."""
        for epoch in range(self.config.num_epochs):
            self._train_epoch(epoch)

            if self.accelerator.is_main_process:
                self._save_and_push_model(epoch)

        if self.accelerator.is_main_process:
            self._save_losses()

    def _train_epoch(self, epoch):
        """Train for one epoch."""
        progress_bar = tqdm(total=len(self.train_dataloader), disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        epoch_loss = 0

        for step, batch in enumerate(self.train_dataloader):
            clean_images = batch.float()
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]
            timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()

            noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)

            with self.accelerator.accumulate(self.model):
                noise_pred = self.model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                self.accelerator.backward(loss)
                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

            epoch_loss += loss.item()
            progress_bar.update(1)

            logs = {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0], "step": self.global_step}
            progress_bar.set_postfix(**logs)
            self.accelerator.log(logs, step=self.global_step)
            self.global_step += 1

        self.epoch_losses.append(epoch_loss / len(self.train_dataloader))

    def _save_and_push_model(self, epoch):
        """Save and push the model to Hugging Face Hub."""
        from diffusers import DDPMPipeline  # Ensure the correct import for your pipeline

        pipeline = DDPMPipeline(unet=self.accelerator.unwrap_model(self.model), scheduler=self.noise_scheduler)

        if (epoch + 1) % self.config.save_image_epochs == 0 or epoch == self.config.num_epochs - 1:
            self._evaluate_model(epoch, pipeline)

        if (epoch + 1) % self.config.save_model_epochs == 0 or epoch == self.config.num_epochs - 1:
            pipeline.save_pretrained(self.config.output_dir)
            if self.config.push_to_hub:
                self.api.upload_folder(repo_id=self.repo_name, folder_path=self.config.output_dir, token=self.token)

    def _evaluate_model(self, epoch, pipeline):
        """Evaluation and image saving logic."""
        # Placeholder: Replace with your evaluation logic
        pass

    def _save_losses(self):
        """Save the epoch losses to a file."""
        losses_path = os.path.join(self.config.output_dir, "losses.txt")
        with open(losses_path, "w") as f:
            for loss in self.epoch_losses:
                f.write(f"{loss}\n")
