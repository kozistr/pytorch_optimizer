import os

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from pytorch_optimizer import SophiaH


class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        x, _ = batch
        x = x.view(x.size(0), -1)

        z = self.encoder(x)
        x_hat = self.decoder(z)

        loss = nn.functional.mse_loss(x_hat, x)

        self.manual_backward(loss, create_graph=True)
        opt.step()

        self.log('train_loss', loss)

        return loss

    def configure_optimizers(self):
        return SophiaH(self.parameters())


def main():
    train_dataset = MNIST(os.getcwd(), train=True, download=True, transform=ToTensor())
    train_loader = DataLoader(train_dataset)

    autoencoder = LitAutoEncoder()
    autoencoder.train()

    if torch.cuda.is_available():
        autoencoder.cuda()

    trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)


if __name__ == '__main__':
    main()
