import os

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from pytorch_optimizer import Lookahead


class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)

        z = self.encoder(x)
        x_hat = self.decoder(z)

        loss = nn.functional.mse_loss(x_hat, x)

        self.log('train_loss', loss)

        return loss

    def configure_optimizers(self):
        return Lookahead(AdamW(self.parameters(), lr=1e-3), k=5, alpha=0.5)


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
