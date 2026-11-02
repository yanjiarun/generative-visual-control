import json

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from supplementary.visual_robustness.common import (
    BASE_SEED, BATCH_SIZE, CONDITIONS, CORRUPTION_SPEC, MODEL_PATHS, OFFLINE_COUNT,
    OFFLINE_DIR, REPETITIONS, ROOT, RUN_DIR, corrupt_image, corruption_seed,
    model_fingerprints, write_json,
)
from src.baseline.cnn_model import BaselineCNN
from src.baseline.position_localization import CNNPositionRegressor
from src.generative_model.core.generative_controller import FlowMatchingController, sample_action
from src.path_following.create_path_cassini_oval import compute_cassini_field_with_gradients

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PREPROCESS = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop((224, 224)), transforms.ToTensor()])
METHODS = ("gvc", "cnn_based", "cnn_localization_gvf")


def tensor(image):
    return PREPROCESS(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))


def angular_errors(pred, truth):
    pn, tn = np.linalg.norm(pred, axis=1), np.linalg.norm(truth, axis=1)
    valid = np.isfinite(pred).all(axis=1) & (pn > 1e-8)
    errors = np.abs(np.degrees(np.arctan2(pred[:, 0] * truth[:, 1] - pred[:, 1] * truth[:, 0],
                                        np.sum(pred * truth, axis=1))))
    errors[~valid] = 180.0  # Penalize invalid predictions.
    errors[tn <= 1e-8] = np.nan  # Exclude undefined reference directions.
    return errors, valid


def load_models():
    models = (FlowMatchingController(history_length=1), BaselineCNN(), CNNPositionRegressor())
    for method, model in zip(METHODS, models):
        checkpoint = torch.load(ROOT / MODEL_PATHS[method], map_location="cpu")
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        model.to(DEVICE).eval().requires_grad_(False)
    return models


def main():
    manifest = OFFLINE_DIR / "heldout_manifest.csv"
    if not manifest.is_file():
        raise SystemExit("First run with Unity: python -m supplementary.visual_robustness.collect_heldout")
    table = pd.read_csv(manifest)
    if len(table) != OFFLINE_COUNT:
        raise ValueError(f"Expected {OFFLINE_COUNT} held-out observations")
    images = [cv2.imread(str(OFFLINE_DIR / "images" / name)) for name in table.image]
    if any(im is None for im in images):
        raise FileNotFoundError("Held-out collection incomplete; run collect_heldout.py first")
    xy = table[["x_norm", "y_norm"]].to_numpy()
    truth = np.array([compute_cassini_field_with_gradients(*point)[:2] for point in xy])
    out = RUN_DIR / "offline"
    out.mkdir(parents=True, exist_ok=True)
    protocol = {"models": model_fingerprints(), "corruption": CORRUPTION_SPEC, "base_seed": BASE_SEED,
                "ode_steps": 50, "sampling": json.loads((OFFLINE_DIR / "sampling_config.json").read_text(encoding="utf-8"))}
    path = out / "protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Offline protocol changed; choose a new RUN_TAG")
    write_json(path, protocol)
    gvc, cnn, loc = load_models()
    for repetition in range(REPETITIONS):
        path = out / f"errors_rep_{repetition:02d}.csv"
        if path.exists():
            continue
        rng = torch.Generator(device=DEVICE).manual_seed(BASE_SEED + repetition)
        latent = torch.rand(len(images), gvc.action_dim, device=DEVICE, generator=rng) * 2 - 1
        rows = []
        for condition in CONDITIONS:
            outputs = {method: [] for method in METHODS}
            for start in range(0, len(images), BATCH_SIZE):
                batch = torch.stack([tensor(corrupt_image(im, condition, corruption_seed(repetition, i + 1, condition)))
                                     for i, im in enumerate(images[start:start + BATCH_SIZE], start)]).to(DEVICE)
                with torch.inference_mode():
                    predicted = sample_action(gvc, batch.unsqueeze(1), num_steps=50, device=DEVICE,
                                              initial_actions=latent[start:start + len(batch)]).cpu().numpy()
                    cnn_output = cnn(batch).cpu().numpy()
                    positions = loc(batch).cpu().numpy()
                outputs["gvc"].extend(predicted[:, :2])
                outputs["cnn_based"].extend(cnn_output[:, :2])
                outputs["cnn_localization_gvf"].extend(
                    compute_cassini_field_with_gradients(*point)[:2] if np.isfinite(point).all() else [np.nan, np.nan]
                    for point in positions)
            for method, values in outputs.items():
                values = np.asarray(values, float)
                errors, valid = angular_errors(values, truth)
                for index, error in enumerate(errors):
                    rows.append({"method": method, "condition": condition, "repetition": repetition,
                                 "sample": index + 1, "true_ux": truth[index, 0], "true_uy": truth[index, 1],
                                 "pred_ux": values[index, 0], "pred_uy": values[index, 1],
                                 "valid_prediction": valid[index], "angular_error_deg": error})
            print(f"Offline rep {repetition} / {condition}")
        pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Offline results: {out}")


if __name__ == "__main__":
    main()
