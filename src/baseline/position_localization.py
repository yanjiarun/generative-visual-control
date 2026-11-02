import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
METERS_PER_UNIT = 100.0  # world_xy = [576, 275] + 100 * normalized_xy
MODEL_VARIANT = "cnn256"
CSV_PATH = ROOT / "src/path_following/vector_field_with_gradients_cassini.csv"
IMAGE_DIR = ROOT / "src/path_following/rectified_panorama_cassini"
SAVE_DIR = ROOT / "src/baseline/models/position_cassini_cnn"
MODEL_PATH = SAVE_DIR / "position_best.pth"
EPOCHS, BATCH_SIZE, LEARNING_RATE, PATIENCE = 120, 32, 1e-3, 20
SEED, WORKERS = 42, 0


class UnityPositionDataset(Dataset):
    def __init__(self, csv_path, image_dir):
        self.rows = pd.read_csv(csv_path)
        required = {"frame_id", "x", "y"}
        if not required.issubset(self.rows.columns):
            raise ValueError(f"CSV must contain columns {sorted(required)}")
        self.image_dir = Path(image_dir)
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        image_path = self.image_dir / f"rectified_color_panorama_{int(row.frame_id):04d}.jpg"
        with Image.open(image_path) as image:
            image = self.preprocess(image.convert("RGB"))
        target = torch.tensor([row.x, row.y], dtype=torch.float32)
        return image, target


class CNNPositionRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.history_length = 1
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten())
        self.regressor = nn.Sequential(
            nn.Linear(64 * 28 * 28, 256), nn.ReLU(), nn.Linear(256, 2))

    def forward(self, images):
        return self.regressor(self.features(images))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum, distance_sum, count = 0.0, 0.0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        delta = model(images) - targets
        loss_sum += delta.square().sum().item()
        distance_sum += torch.linalg.vector_norm(delta, dim=1).sum().item()
        count += len(targets)
    return (loss_sum / (2 * count), METERS_PER_UNIT * distance_sum / count,
            METERS_PER_UNIT * (loss_sum / count) ** 0.5)


def main(train=True, epochs=EPOCHS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = None if train else torch.load(MODEL_PATH, map_location=device)
    seed = SEED if train else checkpoint.get("seed", SEED)
    torch.manual_seed(seed)
    dataset = UnityPositionDataset(CSV_PATH, IMAGE_DIR)
    if len(dataset) < 1:
        raise ValueError("the training dataset is empty")
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=WORKERS, pin_memory=device.type == "cuda")
    fit_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=WORKERS, pin_memory=device.type == "cuda")
    model = CNNPositionRegressor().to(device)
    phase = "training on all grid images" if train else "evaluating all grid images"
    print(f"Device: {device}; {phase}: {len(dataset)}; seed={seed}")

    if not train:
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=6, min_lr=1e-5)
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        best_fit = float("inf")
        stale_epochs = 0
        history = []
        for epoch in range(1, epochs + 1):
            model.train()
            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(model(images), targets)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            fit_mse, fit_mae, fit_rmse = evaluate(model, fit_loader, device)
            row = {"epoch": epoch, "full_grid_mse": fit_mse,
                   "full_grid_position_mae_m": fit_mae, "full_grid_position_rmse_m": fit_rmse,
                   "learning_rate": optimizer.param_groups[0]["lr"]}
            history.append(row)
            (SAVE_DIR / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            print(f"Epoch {epoch:03d}/{epochs}: full_grid_mse={fit_mse:.6f}, "
                  f"position_MAE={fit_mae:.4f} m, position_RMSE={fit_rmse:.4f} m")
            if fit_mse < best_fit - 1e-8:
                best_fit = fit_mse
                stale_epochs = 0
                torch.save({
                    "model_state_dict": model.state_dict(), "epoch": epoch,
                    "full_grid_mse": fit_mse, "history_length": model.history_length,
                    "seed": seed, "training_samples": len(dataset),
                    "meters_per_unit": METERS_PER_UNIT,
                    "model_variant": MODEL_VARIANT,
                }, MODEL_PATH)
            else:
                stale_epochs += 1
            scheduler.step(fit_mse)
            if stale_epochs >= PATIENCE:
                print(f"Stopping after {epoch} epochs without a new full-grid best.")
                break

        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    fit_mse, fit_mae, fit_rmse = evaluate(model, fit_loader, device)
    training_samples = checkpoint.get("training_samples")
    trained_on_all = training_samples == len(dataset) if training_samples is not None else None
    metric_scope = ("training fit; not held-out validation" if trained_on_all else
                    "full-grid evaluation; checkpoint training coverage is unconfirmed")
    print(f"Full-grid position error (fit diagnostic, not held-out validation): "
          f"MAE={fit_mae:.4f} m, RMSE={fit_rmse:.4f} m "
          f"(normalized coordinate MSE={fit_mse:.6f}; n={len(dataset)})")
    metrics = {"model_path": str(MODEL_PATH), "model_variant": MODEL_VARIANT,
               "samples": len(dataset), "training_samples": training_samples, "seed": seed,
               "trained_on_all_grid_images": trained_on_all,
               "full_grid_normalized_mse": fit_mse, "full_grid_position_mae_m": fit_mae,
               "full_grid_position_rmse_m": fit_rmse, "meters_per_unit": METERS_PER_UNIT,
               "metric_scope": metric_scope}
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = SAVE_DIR / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Training metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate the position CNN.")
    parser.add_argument("--evaluate", action="store_true", help="evaluate the saved position checkpoint")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    main(train=not args.evaluate, epochs=args.epochs)
