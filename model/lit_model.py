import pytorch_lightning as pl
import torchmetrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchsummary import summary

# import PSNR and SSIM metrics from torchmetrics
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from .arch import dyconvsr


class LitDyConvSR(pl.LightningModule):
    def __init__(
        self, 
        config
    ):
        super().__init__()
        self.config = config
        self.lr = config.learning_rate
        self.model = dyconvsr(config)

        self.loss_fn = nn.L1Loss()
        
        # add metrics to monitor during training
        self.train_psnr = PeakSignalNoiseRatio()
        self.val_psnr = PeakSignalNoiseRatio()
        self.train_ssim = StructuralSimilarityIndexMeasure()
        self.val_ssim = StructuralSimilarityIndexMeasure()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        
    def forward(self, x):
        return self.model(x)

    def step(self, batch, batch_idx):
        image_lr, image_hr = batch['lr'], batch['hr']
        image_lr = image_lr.to(self.device)
        image_sr = self.forward(image_lr)
        
        loss = self.loss_fn(image_sr, image_hr)
        
        psnr = self.train_psnr(image_sr, image_hr)
        ssim = self.train_ssim(image_sr, image_hr)
        
        return loss, psnr, ssim, image_sr, image_hr
    
    
    def training_step(self, batch, batch_idx):
        loss, psnr, ssim, image_sr, image_hr = self.step(batch, batch_idx)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_psnr", psnr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_ssim", ssim, on_step=False, on_epoch=True, prog_bar=True)
        
        # log images to wandb log 4 times per epoch
        if batch_idx % 500 == 0:
            grid = torchvision.utils.make_grid(torch.cat((image_sr[:1], image_hr[:1]), dim=0))
            self.logger.experiment.add_image('train_images', grid, self.global_step)
        
        return {"loss": loss, "psnr": psnr, "ssim": ssim}
    
    def validation_step(self, batch, batch_idx):
        loss, psnr, ssim, image_sr, image_hr = self.step(batch, batch_idx)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_psnr", psnr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_ssim", ssim, on_step=False, on_epoch=True, prog_bar=True)
        
        if batch_idx % 500 == 0:
            # show random single images lr, sr, hr in tensorboard
            index = torch.randint(0, image_sr.shape[0], (1,))
            grid = torchvision.utils.make_grid(torch.cat((image_sr[index:index+1], image_hr[index:index+1]), dim=0))
            self.logger.experiment.add_image('val_images', grid, self.global_step)
        
        return {"loss": loss, "psnr": psnr, "ssim": ssim}
    
    def test_step(self, batch, batch_idx):
        loss, psnr, ssim = self.step(batch, batch_idx)
        
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_psnr", psnr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_ssim", ssim, on_step=False, on_epoch=True, prog_bar=True)
        
        return {"loss": loss, "psnr": psnr, "ssim": ssim}
    
    def predict_step(self, image_lr):
        image_sr = self.forward(image_lr)
        return image_sr
    
    def configure_optimizers(self):
        if self.config.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5, 
                                          betas=(0.9, 0.9999), amsgrad=False)
        elif self.config.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5,
                                         betas=(0.9, 0.9999), amsgrad=False)
        elif self.config.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)
        else:
            raise NotImplementedError("Optimizer not implemented")
        
        # create learning rate scheduler with halved for every 200000 steps
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.config.multistepLR_milestones, 
            gamma=self.config.multistepLR_gamma, 
            verbose=False,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": self.config.lr_monitor_logging_interval,
                "frequency": 1,
            },
        }
    
