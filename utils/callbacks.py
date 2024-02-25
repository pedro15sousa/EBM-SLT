import torch
import torchvision
# PyTorch Lightning
import pytorch_lightning as pl
# Callbacks
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
# Pytorch Summary
from torchsummary import summary

import numpy as np
import random

from metrics.scores import inception_score, frechet_inception_distance
from utils.sampler import Sampler

# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "./saved_models"

class InceptionScoreCallback(pl.Callback):
    def __init__(self, mnist_classifier):
        """
        Callback to compute the Inception Score at the end of each epoch.

        Args:
            mnist_classifier (torch.nn.Module): Pre-trained classifier model to use for Inception Score.
        """
        self.mnist_classifier = mnist_classifier
        self.scores = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Called at the end of the training batch to compute metrics.
        """
        real_imgs, _ = batch
        fake_imgs = pl_module.sample_images_for_metrics()  # Assuming a method to generate fake images

        # Compute and store metrics
        score = self.compute_inception_score(fake_imgs)
        self.scores.append(score)
        self.log('inception_score', score)

    def on_train_epoch_end(self, trainer, pl_module, unused=None):
        """
        Called at the end of the training epoch to compute mean of metrics.
        """
        mean_score = np.mean(self.scores)

        # Log the mean scores
        trainer.logger.log_metrics({'mean_inception_score': mean_score}, step=trainer.current_epoch)

        print("\n------------Inception---------------")
        print("Mean Inception Score: ", mean_score)
        print("------------------------------------\n")

        # Reset metrics list for the next epoch
        self.scores = []

    def compute_inception_score(self, samples_energ):
        """
        Compute the Inception Score for the given samples.

        Args:
            samples_energ (torch.Tensor): Tensor containing the generated samples.

        Returns:
            float: Computed Inception Score.
        """
        with torch.no_grad():
            self.mnist_classifier.eval()  # Ensure the classifier is in eval mode
            log_p_yx = self.mnist_classifier(samples_energ)
            p_yx = torch.exp(log_p_yx).cpu().numpy()
            # print("Samples Energy Shape: ", samples_energ.shape)
            # print("p_yx Shape: ", p_yx.cpu().numpy().shape)
            score = inception_score(p_yx)  # Assuming inception_score accepts numpy array
            # print(score)
        return score


class FIDCallback(pl.Callback):
    def __init__(self, mnist_classifier):
        """
        Callback to compute the FID Score at the end of each epoch.

        Args:
            mnist_classifier (torch.nn.Module): Pre-trained classifier model to use for FID Score.
        """
        self.mnist_classifier = mnist_classifier
        self.scores = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Called at the end of the training batch to compute metrics.
        """
        real_imgs, _ = batch
        fake_imgs = pl_module.sample_images_for_metrics()  # Assuming a method to generate fake images
        # Compute and store metrics
        score = self.compute_fid_score(real_imgs, fake_imgs)
        self.scores.append(score)
        self.log('fid_score', score)

    def on_train_epoch_end(self, trainer, pl_module, unused=None):
        """
        Called at the end of the training epoch to compute mean of metrics.
        """
        mean_score = np.mean(self.scores)

        # Log the mean scores
        trainer.logger.log_metrics({'mean_fid_score': mean_score}, step=trainer.current_epoch)

        print("\n---------------FID------------------")
        print("Mean FID Score: ", mean_score)
        print("------------------------------------\n")

        # Reset metrics list for the next epoch
        self.scores = []


    def compute_fid_score(self, samples_data, samples_energ):
        with torch.no_grad():
            stats_gen = self.mnist_classifier.get_activations(samples_energ).cpu().numpy()
            stats_real = self.mnist_classifier.get_activations(samples_data).cpu().numpy()
            score = frechet_inception_distance(stats_real, stats_gen)
            # print(stats_gen.shape)
            # print(stats_real.shape)
            # print(score)
        return - score


class GenerateImagesCallback(pl.Callback):

    def __init__(self, batch_size=8, vis_steps=8, num_steps=256, every_n_epochs=5):
        super().__init__()
        self.batch_size = batch_size         # Number of images to generate
        self.vis_steps = vis_steps           # Number of steps within generation to visualize
        self.num_steps = num_steps           # Number of steps to take during generation
        self.every_n_epochs = every_n_epochs # Only save those images every N epochs (otherwise tensorboard gets quite large)

    def on_train_epoch_end(self, trainer, pl_module, unused=None):
        # Skip for all other epochs
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Generate images
            imgs_per_step = self.generate_imgs(pl_module)
            # Plot and add to tensorboard
            for i in range(imgs_per_step.shape[1]):
                step_size = self.num_steps // self.vis_steps
                imgs_to_plot = imgs_per_step[step_size-1::step_size,i]
                # grid = torchvision.utils.make_grid(imgs_to_plot, nrow=imgs_to_plot.shape[0], normalize=True, range=(-1,1))
                
                # Normalize the images
                imgs_to_plot = (imgs_to_plot + 1) / 2  # Rescale images from [-1, 1] to [0, 1]
                grid = torchvision.utils.make_grid(imgs_to_plot, nrow=imgs_to_plot.shape[0], normalize=True)
                
                trainer.logger.experiment.add_image(f"generation_{i}", grid, global_step=trainer.current_epoch)

    def generate_imgs(self, pl_module):
        pl_module.eval()
        start_imgs = torch.rand((self.batch_size,) + pl_module.hparams["img_shape"]).to(pl_module.device)
        start_imgs = start_imgs * 2 - 1
        torch.set_grad_enabled(True)  # Tracking gradients for sampling necessary
        imgs_per_step = Sampler.generate_samples(pl_module.cnn, start_imgs, pl_module.slt, steps=self.num_steps, step_size=10, return_img_per_step=True)
        torch.set_grad_enabled(False)
        pl_module.train()
        return imgs_per_step
    
class SamplerCallback(pl.Callback):

    def __init__(self, num_imgs=32, every_n_epochs=5):
        super().__init__()
        self.num_imgs = num_imgs             # Number of images to plot
        self.every_n_epochs = every_n_epochs # Only save those images every N epochs (otherwise tensorboard gets quite large)

    def on_train_epoch_end(self, trainer, pl_module, unused=None):
        if trainer.current_epoch % self.every_n_epochs == 0:
            exmp_imgs = torch.cat(random.choices(pl_module.sampler.examples, k=self.num_imgs), dim=0)
            # grid = torchvision.utils.make_grid(exmp_imgs, nrow=4, normalize=True, range=(-1,1))

            # Normalize the images
            exmp_imgs = (exmp_imgs + 1) / 2  # Rescale images from [-1, 1] to [0, 1]
            grid = torchvision.utils.make_grid(exmp_imgs, nrow=exmp_imgs.shape[0], normalize=True)

            trainer.logger.experiment.add_image("sampler", grid, global_step=trainer.current_epoch)


class OutlierCallback(pl.Callback):

    def __init__(self, batch_size=1024):
        super().__init__()
        self.batch_size = batch_size

    def on_train_epoch_end(self, trainer, pl_module, unused=None):
        with torch.no_grad():
            pl_module.eval()
            rand_imgs = torch.rand((self.batch_size,) + pl_module.hparams["img_shape"]).to(pl_module.device)
            rand_imgs = rand_imgs * 2 - 1.0
            rand_out = pl_module.cnn(rand_imgs).mean()
            pl_module.train()

        trainer.logger.experiment.add_scalar("rand_out", rand_out, global_step=trainer.current_epoch)

