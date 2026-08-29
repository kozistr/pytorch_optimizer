"""Example: training with ExponentialScheduler.

This example shows how to use the newly added ExponentialScheduler together with
an optimizer from pytorch_optimizer.

Usage:
    python examples/exponential_scheduler_example.py
"""

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pytorch_optimizer import AdamW
from pytorch_optimizer.lr_scheduler import ExponentialScheduler


class SimpleModel(nn.Module):
    def __init__(self, input_dim: int = 32, hidden_dim: int = 64, output_dim: int = 10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def build_data(num_samples: int = 1024, input_dim: int = 32, num_classes: int = 10):
    x = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(x, y)


def train(epochs: int = 5, batch_size: int = 64, lr: float = 1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SimpleModel().to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = ExponentialScheduler(
        optimizer,
        t_max=epochs * (1024 // batch_size),
        max_lr=lr,
        min_lr=1e-5,
        init_lr=lr * 0.1,
        warmup_steps=5,
        gamma=0.95,
    )

    dataset = build_data()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * inputs.size(0)

        avg_loss = epoch_loss / len(dataset)
        print(
            f'Epoch {epoch + 1}/{epochs} - '
            f'loss: {avg_loss:.4f} - '
            f'lr: {scheduler.get_lr():.6f}'
        )


def parse_args():
    parser = argparse.ArgumentParser(description='Run exponential scheduler example.')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
