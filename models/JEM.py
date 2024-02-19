import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn.utils.prune as prune

from utils.sampler import Sampler
from energy_funcs.resnet import ResNet18

class JEM(pl.LightningModule):

    def __init__(self, img_shape, batch_size, n_classes=10, alpha=0.1, lr=1e-4, beta1=0.0, f=ResNet18, **f_args):
        super().__init__()
        self.save_hyperparameters()

        self.cnn = f(**f_args)
        # self.class_output = nn.Linear(self.cnn.last_dim, n_classes)
        self.sampler = Sampler(self.cnn, img_shape=img_shape, sample_size=batch_size)
        self.example_input_array = torch.zeros(1, *img_shape)
        self.weight_ce_loss = torch.tensor(1.0)

    def apply_unstructured_pruning(self, amount=0.5):
        """Applies unstructured L1 pruning to each Conv2d layer in the resnet."""
        for module in self.cnn.modules():
            if isinstance(module, nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=amount)
                prune.remove(module, 'weight')

    def apply_structured_pruning(self, amount=0.5):
        """Applies structured L1 pruning to each Conv2d layer in the resnet."""
        for module in self.cnn.modules():
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(module, name='weight', amount=amount, n=1, dim=0)  # dim=0 for filter pruning
                prune.remove(module, 'weight')

    def print_pruning_stats(self):
        total_params = 0
        pruned_params = 0
        for param in self.parameters():
            # Count all parameters and pruned parameters (those set to zero)
            total_params += param.numel()
            pruned_params += (param == 0).sum().item()
        print(f"Total parameters: {total_params}")
        print(f"Pruned parameters: {pruned_params}")

    def compute_accuracy(self, logits, labels):
        correct = logits.argmax(dim=1).eq(labels).sum().item()
        total = labels.size(0)
        return correct / total

    def classify(self, x):
        penult_z = self.cnn(x)
        return penult_z

    def sample_images_for_metrics(self):
        # Generate images for computing metrics
        samples = self.sampler.sample_new_exmps(steps=60, step_size=10)
        return samples

    def forward(self, x, y=None):
        logits = self.classify(x)

        if logits.dim() == 1:
            return logits.logsumexp(0)

        if y is None:
            return logits.logsumexp(1)
        else:
            return torch.gather(logits, 1, y[:, None])

    def configure_optimizers(self):
        # Energy models can have issues with momentum as the loss surfaces changes with its parameters.
        # Hence, we set it to 0 by default.
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.97) # Exponential decay over epochs
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # We add minimal noise to the original images to prevent the model from focusing on purely "clean" inputs
        real_imgs, labels = batch
        small_noise = torch.randn_like(real_imgs) * 0.005
        real_imgs.add_(small_noise).clamp_(min=-1.0, max=1.0)

        # Obtain samples
        fake_imgs = self.sampler.sample_new_exmps(steps=60, step_size=10)

        # Predict energy score for all images
        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        # real_out, fake_out = self.cnn(inp_imgs).chunk(2, dim=0)
        real_out, fake_out = self.forward(inp_imgs).chunk(2, dim=0)

        # Calculate losses
        reg_loss = self.hparams.alpha * (real_out ** 2 + fake_out ** 2).mean()
        cdiv_loss = fake_out.mean() - real_out.mean()
        energy_loss = reg_loss + cdiv_loss

        # Classification loss
        pred = self.classify(real_imgs)
        acc = self.compute_accuracy(pred, labels)
        class_loss = F.cross_entropy(pred, labels)

        # Total Loss
        loss = cdiv_loss + class_loss * self.weight_ce_loss

        # Logging
        # Logging
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('loss_regularization', reg_loss)
        self.log('loss_contrastive_divergence', cdiv_loss)
        self.log('metrics_avg_real', real_out.mean())
        self.log('metrics_avg_fake', fake_out.mean())
        self.log('train_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # For validating, we calculate the contrastive divergence between purely random images and unseen examples
        # Note that the validation/test step of energy-based models depends on what we are interested in the model
        real_imgs, labels = batch
        pred = self.classify(real_imgs)
        val_acc = self.compute_accuracy(pred, labels)
        val_class_loss = F.cross_entropy(pred, labels)

        fake_imgs = torch.rand_like(real_imgs) * 2 - 1
        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        real_out, fake_out = self.forward(inp_imgs).chunk(2, dim=0)
        cdiv = fake_out.mean() - real_out.mean()

        loss = cdiv + val_class_loss * self.weight_ce_loss
        
        self.log('val_accuracy', val_acc, on_epoch=True, prog_bar=True)
        self.log('val_contrastive_divergence', loss, on_epoch=True, prog_bar=True)
        self.log('val_fake_out', fake_out.mean())
        self.log('val_real_out', real_out.mean())