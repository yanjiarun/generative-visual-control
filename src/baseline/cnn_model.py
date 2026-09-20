import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.generative_model.core.datasets import (
    RectangularMaskedDataset, VectorFieldWithGradientsDataset)


class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten())
        self.regressor = nn.Sequential(
            nn.Linear(64 * 28 * 28, 16), nn.ReLU(), nn.Linear(16, 4))

    def forward(self, x):
        return self.regressor(self.features(x))


def train_cnn_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    log_dir = os.path.join(args.save_dir,
                           f"logs_{datetime.now():%Y%m%d_%H%M%S}")
    writer = SummaryWriter(log_dir)
    dataset_cls = (RectangularMaskedDataset if args.dataset_type == "masked"
                   else VectorFieldWithGradientsDataset)
    kwargs = {"mask_type": args.mask_type} if args.dataset_type == "masked" else {}
    dataset = dataset_cls(args.vector_field_file, args.img_dir, **kwargs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = BaselineCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for images, targets in loader:
            images, targets = images.squeeze(1).to(device), targets.to(device)
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(images), targets)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total += loss.item()
        avg = total / len(loader)
        writer.add_scalar("Loss/train", avg, epoch + 1)
        print(f"Epoch [{epoch + 1}/{args.epochs}], Average Loss: {avg:.4f}")
        if (epoch + 1) % args.checkpoint_interval == 0:
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "loss": avg},
                       os.path.join(args.save_dir,
                                    f"cnn_checkpoint_epoch_{epoch + 1}.pth"))
        if avg < best:
            best = avg
            torch.save(model.state_dict(), os.path.join(args.save_dir, "cnn_best.pth"))
    torch.save(model.state_dict(), os.path.join(args.save_dir, "cnn_final.pth"))
    writer.close()
    return best


if __name__ == "__main__":
    class Args:
        pass

    a = Args()
    a.batch_size, a.epochs, a.learning_rate = 32, 300, 1e-3
    a.vector_field_file = "../path_following/vector_field_with_gradients_quartic.csv"
    a.img_dir = "../path_following/rectified_panorama_quartic"
    a.save_dir = "./models/cnn_quartic_models"
    a.grad_clip, a.checkpoint_interval = 1.0, 50
    a.dataset_type, a.mask_type = "normal", "none"
    train_cnn_model(a)
