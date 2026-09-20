import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib.patches import Ellipse, Patch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.baseline.cnn_model import BaselineCNN
from src.generative_model.core.diffusion_policy import DiffusionPolicy
from src.generative_model.core.generative_controller import (
    FlowMatchingController, sample_action)
from src.path_following.create_path_quartic_obstacle import (
    BRANCH_X, DECISION_X, OBSTACLE_CENTER, OBSTACLE_RADII, X_RANGE,
    Y_RANGE, route)


ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "src/path_following/vector_field_quartic_obstacle.csv"
IMAGES = ROOT / "src/path_following/rectified_panorama_quartic_obstacle"
WEIGHTS = {
    "gvc": ROOT / "src/generative_model/models/flow_matching_quartic_obstacle_models/flow_matching_policy_final.pth",
    "cnn": ROOT / "src/baseline/models/cnn_quartic_obstacle_models/cnn_final.pth",
    "dp": ROOT / "src/generative_model/models/diffusion_policy_quartic_obstacle_models/diffusion_policy_final.pth",
}
COLORS = {-1: "#D55E00", 0: "0.45", 1: "#0072B2"}


def target4(table):
    table = table.copy()
    table["ex"] = table.grad_uy_dx * table.ux - table.grad_ux_dx * table.uy
    table["ey"] = table.grad_uy_dy * table.ux - table.grad_ux_dy * table.uy
    return table, table[["ux", "uy", "ex", "ey"]].to_numpy(float)


class GridImages(Dataset):
    def __init__(self, table, image_dir, stride=2):
        points = table.drop_duplicates("frame_id").copy()
        xs = {x: i for i, x in enumerate(sorted(points.x.unique()))}
        ys = {y: i for i, y in enumerate(sorted(points.y.unique()))}
        keep = [xs[x] % stride == 0 and ys[y] % stride == 0
                for x, y in zip(points.x, points.y)]
        self.points = points.loc[keep].reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 256)), transforms.CenterCrop((224, 224))])

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):
        frame = int(self.points.iloc[index].frame_id)
        image = self.preprocess(Image.open(self.image_dir /
            f"rectified_color_panorama_{frame:04d}.jpg").convert("RGB"))
        image = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255
        return image.unsqueeze(0)


def load_model(method, path, device):
    model = {"gvc": FlowMatchingController,
             "cnn": BaselineCNN,
             "dp": DiffusionPolicy}[method]().to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state.get("model_state_dict", state))
    return model.eval()


def infer(method, model, loader, samples, ode_steps, dp_steps, device):
    repeats = 1 if method == "cnn" else samples
    chunks = []
    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            if method == "gvc":
                output = torch.stack([sample_action(
                    model, images, ode_steps, device) for _ in range(repeats)])
            elif method == "dp":
                output = torch.stack([model.predict_action(
                    images, dp_steps, deterministic=True)[:, 0]
                                      for _ in range(repeats)])
            else:
                output = model(images[:, 0]).unsqueeze(0)
            chunks.append(output.cpu().numpy())
    return np.concatenate(chunks, axis=1)


def unit(vectors):
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-8)


def evaluate(prediction, targets, modes):
    direction = unit(prediction[..., :2])
    assigned = np.empty(direction.shape[:2], int)
    angles = np.empty(direction.shape[:2])
    gradients = []
    for i, (truth, labels) in enumerate(zip(targets, modes)):
        nearest = np.argmax(direction[:, i] @ unit(truth[:, :2]).T, axis=1)
        assigned[:, i] = labels[nearest]
        dots = np.sum(direction[:, i] * unit(truth[nearest, :2]), axis=1)
        angles[:, i] = np.degrees(np.arccos(np.clip(dots, -1, 1)))
        if prediction.shape[-1] >= 4:
            gradients.extend(np.mean((prediction[:, i, 2:4] - truth[nearest, 2:4]) ** 2, axis=1))
    bimodal = np.array([len(x) > 1 for x in modes])
    recovered = [len(np.unique(assigned[:, i])) == 2 for i in np.flatnonzero(bimodal)]
    row = {
        "mean_angle_deg": angles.mean(),
        "deterministic_angle_deg": angles[:, ~bimodal].mean(),
        "bimodal_angle_deg": angles[:, bimodal].mean(),
        "magnitude_mae": np.mean(np.abs(np.linalg.norm(prediction[..., :2], axis=-1) - 1)),
        "gradient_rmse": np.sqrt(np.mean(gradients)) if gradients else np.nan,
        "upper_sample_ratio": np.mean(assigned[:, bimodal] == 1),
        "both_modes_recovered_pct": 100 * np.mean(recovered),
    }
    return row, assigned


def decorate(ax):
    xx = np.linspace(*BRANCH_X, 400)
    for mode, style in ((0, "--"), (1, "-"), (-1, "-")):
        ax.plot(xx, route(xx, mode)[0], style, color=COLORS[mode], lw=1)
    ax.add_patch(Ellipse(OBSTACLE_CENTER, 2 * OBSTACLE_RADII[0],
                         2 * OBSTACLE_RADII[1], color="#CC79A7", alpha=.22))
    ax.axvspan(*DECISION_X, color="#CC79A7", alpha=.08)
    ax.set(xlim=X_RANGE, ylim=Y_RANGE, aspect="equal", xlabel="x", ylabel="y")
    ax.grid(alpha=.2)


def plot_fields(table, points, predictions, assignments, save):
    fig, axes = plt.subplots(1, len(predictions) + 1,
                             figsize=(5 * (len(predictions) + 1), 5))
    for mode in (0, 1, -1):
        rows = table[table["mode"] == mode]
        axes[0].quiver(rows.x, rows.y, rows.ux, rows.uy, color=COLORS[mode],
                       angles="xy", scale_units="xy", scale=50, alpha=.65)
    axes[0].set_title("Training labels")
    decorate(axes[0])
    for ax, (method, raw) in zip(axes[1:], predictions.items()):
        vectors = unit(raw[..., :2])
        for mode in (0, 1, -1):
            sample, point = np.where(assignments[method] == mode)
            ax.quiver(points.x.iloc[point], points.y.iloc[point],
                      vectors[sample, point, 0], vectors[sample, point, 1],
                      color=COLORS[mode], angles="xy", scale_units="xy",
                      scale=50, alpha=.65 if len(raw) == 1 else .28)
        ax.set_title(f"{method.upper()} ({len(raw)} sample{'s' if len(raw) > 1 else ''})")
        decorate(ax)
    axes[0].legend(handles=[Patch(color=COLORS[m], label=n) for m, n in
                   ((1, "upper mode"), (-1, "lower mode"), (0, "original quartic"))],
                   fontsize=8)
    fig.tight_layout()
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main(args):
    table, _ = target4(pd.read_csv(args.csv))
    dataset = GridImages(table, args.images, args.stride)
    selected = table[table.frame_id.isin(dataset.points.frame_id)]
    groups = [selected[selected.frame_id == frame] for frame in dataset.points.frame_id]
    targets = [target4(group)[1] for group in groups]
    modes = [group["mode"].to_numpy(int) for group in groups]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    loader = DataLoader(dataset, args.batch_size, shuffle=False)
    predictions, assignments, rows = {}, {}, []
    for method in args.methods:
        print(f"Evaluating {method} on {len(dataset)} positions...")
        prediction = infer(method, load_model(method, args.weights[method], device),
                           loader, args.samples, args.ode_steps, args.dp_steps, device)
        metrics, assignments[method] = evaluate(prediction, targets, modes)
        predictions[method] = prediction
        rows.append({"method": method, **metrics})
    save = Path(args.save)
    plot_fields(selected, dataset.points, predictions, assignments, save)
    pd.DataFrame(rows).to_csv(save.with_name(save.stem + "_metrics.csv"), index=False)
    np.savez_compressed(save.with_name(save.stem + "_predictions.npz"), **predictions)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved {save}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", choices=tuple(WEIGHTS), default=tuple(WEIGHTS))
    parser.add_argument("--csv", type=Path, default=CSV)
    parser.add_argument("--images", type=Path, default=IMAGES)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ode-steps", type=int, default=50)
    parser.add_argument("--dp-steps", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save", type=Path,
                        default=ROOT / "comparison_plots_quartic_obstacle/model_vector_fields.png")
    args = parser.parse_args()
    args.weights = WEIGHTS
    main(args)
