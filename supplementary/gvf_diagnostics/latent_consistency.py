import json

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from supplementary.gvf_diagnostics.settings import (
    BATCH_SIZE, DATA, EPS, FD_STEP_M, MODELS, ODE_STEPS, OUT as SOURCE,
    ROOT, SCALE, angle, fd_gradient, guard, sha256, wrap,
)

OUT = ROOT / "data/gvf_diagnostics/latent/run01"
POINT_COUNT, FIXED_SEED = 30, 20261027
LATENT_SEEDS = tuple(range(FIXED_SEED, FIXED_SEED + 20))
VIEWS = ("center", "h_xp", "h_xm", "h_yp", "h_ym",
         "half_xp", "half_xm", "half_yp", "half_ym", "repeat_center")


def select_points(table):
    centers = table[table.local & table.view.eq("center")].sort_values("sample")
    if len(centers) < POINT_COUNT:
        raise ValueError(f"Need at least {POINT_COUNT} existing neighborhood centers")
    xy = centers[["x_norm", "y_norm"]].to_numpy()
    scaled = xy / [5.6, 2.8]
    chosen = [int(np.argmin(np.linalg.norm(scaled, axis=1)))]
    for _ in range(1, POINT_COUNT):
        distances = np.linalg.norm(scaled[:, None] - scaled[chosen], axis=2).min(axis=1)
        distances[chosen] = -1
        chosen.append(int(np.argmax(distances)))
    return centers.iloc[chosen]["sample"].tolist()


class Predictor:
    def __init__(self, table):
        self.table, self.model = table, None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([transforms.Resize((256, 256)),
                                             transforms.CenterCrop((224, 224)), transforms.ToTensor()])
        self.images = {}

    def latent(self, seed):
        rng = torch.Generator(device=self.device).manual_seed(seed)
        return torch.rand(1, 4, device=self.device, generator=rng) * 2 - 1

    def infer(self, indices, latent):
        from src.generative_model.core.generative_controller import FlowMatchingController, sample_action
        if self.model is None:
            self.model = FlowMatchingController(history_length=1).to(self.device)
            checkpoint = torch.load(MODELS["gvc"], map_location="cpu")
            self.model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
            self.model.eval().requires_grad_(False)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        values = []
        with torch.inference_mode():
            for start in range(0, len(indices), BATCH_SIZE):
                ids = indices[start:start + BATCH_SIZE]
                for index in ids:
                    if index not in self.images:
                        path = DATA / "images" / self.table.loc[index, "image"]
                        image = cv2.imread(str(path))
                        if image is None:
                            raise ValueError(f"Unreadable image: {path}")
                        self.images[index] = self.transform(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
                batch = torch.stack([self.images[index] for index in ids]).to(self.device)
                values.extend(sample_action(self.model, batch.unsqueeze(1), ODE_STEPS, self.device,
                                            initial_actions=latent.expand(len(batch), -1)).cpu().numpy())
        return np.asarray(values)


def resampling_changes(outputs):
    valid = np.isfinite(outputs).all(axis=-1) & (np.linalg.norm(outputs[..., :2], axis=-1) > EPS)
    headings = angle(outputs)
    headings[~valid] = np.nan
    i, j = np.triu_indices(len(outputs), 1)
    return np.degrees(abs(wrap(headings[i] - headings[j]))), valid


def mean_max(values):
    values = np.asarray(values, float)
    valid = values[np.isfinite(values)]
    return (f"Mean={valid.mean():.8g}, Max={valid.max():.8g}, valid={len(valid)}/{values.size}"
            if len(valid) else f"undefined, valid=0/{values.size}")


def main():
    manifest = DATA / "manifest.csv"
    if not manifest.is_file():
        raise SystemExit("Missing neighborhood images. First run: python -m supplementary.gvf_diagnostics.collect")
    table = pd.read_csv(manifest)
    samples = select_points(table)
    groups = []
    for sample in samples:
        group = table[table["sample"].eq(sample)]
        groups.append(dict(zip(group.view, group.index)))
    if any(set(g) != set(VIEWS) for g in groups):
        raise ValueError("Selected centers need both saved stencils and repeated-center images")
    for group in groups:
        xy = table.loc[[group[v] for v in VIEWS], ["x_world", "y_world"]].to_numpy()
        expected = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1],
                             [.5, 0], [-.5, 0], [0, .5], [0, -.5], [0, 0]]) * FD_STEP_M
        if not np.allclose(xy - xy[0], expected, atol=1e-8, rtol=0):
            raise ValueError("Saved neighborhood coordinates do not match the configured meter steps")
    indices = [g[view] for g in groups for view in VIEWS]
    centers = [g["center"] for g in groups]
    if len(LATENT_SEEDS) < 2 or LATENT_SEEDS[0] != FIXED_SEED:
        raise ValueError("The fixed seed must be the first of at least two latent draws")
    images = [DATA / "images" / name for name in table.loc[indices, "image"]]
    if any(not p.is_file() for p in images):
        raise SystemExit("Neighborhood collection incomplete; run collect.py again.")
    image_hashes = {p.name: sha256(p) for p in images}
    model_hash = sha256(MODELS["gvc"])
    old_path = SOURCE / "evaluation_protocol.json"
    old = json.loads(old_path.read_text(encoding="utf-8")) if old_path.is_file() else {}
    cache_ok = (old.get("manifest") == sha256(manifest) and old.get("models", {}).get("gvc") == model_hash
                and old.get("ode_steps") == ODE_STEPS
                and all(old.get("images", {}).get(name) == digest for name, digest in image_hashes.items()))
    engine = Predictor(table)
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {"manifest_sha256": sha256(manifest), "images": image_hashes, "model_sha256": model_hash,
                "samples": samples, "selection": "farthest spatial coverage, initial center nearest domain center",
                "latent_seeds": list(LATENT_SEEDS), "fd_step_m": FD_STEP_M,
                "ode_steps": ODE_STEPS, "scale_m": SCALE, "device": str(engine.device),
                "cache_sha256": {str(path): sha256(path) for seed in LATENT_SEEDS
                                 for path in [SOURCE / "predictions" / f"gvc_{seed}.npz"]
                                 if cache_ok and seed in old.get("latent_seeds", []) and path.is_file()},
                "script_sha256": sha256(__file__)}
    guard(OUT / "source_protocol.json", protocol)
    center_outputs, latents, fixed, sources = [], [], None, []
    for seed in LATENT_SEEDS:
        ids = indices if seed == FIXED_SEED else centers
        path = SOURCE / "predictions" / f"gvc_{seed}.npz"
        if cache_ok and seed in old.get("latent_seeds", []) and path.is_file():
            with np.load(path) as saved:
                if saved["outputs"].shape != (len(table), 4) or saved["latent"].shape != (1, 4):
                    raise ValueError(f"Invalid existing prediction cache: {path}")
                latent = torch.as_tensor(saved["latent"], device=engine.device)
                outputs = saved["outputs"][ids]
            sources.append(f"{seed}: reused {path}; sha256={sha256(path)}")
        else:
            latent = engine.latent(seed)
            outputs = engine.infer(ids, latent)
            sources.append(f"{seed}: new inference on saved images")
        latents.append(latent.detach().cpu().numpy()[0].tolist())
        if seed == FIXED_SEED:
            fixed = outputs.reshape(POINT_COUNT, len(VIEWS), 4)
            center_outputs.append(fixed[:, 0])
        else:
            center_outputs.append(outputs)
        print(f"Latent {seed}: {'cached' if 'reused' in sources[-1] else 'inferred'}", flush=True)

    latent = torch.tensor([latents[0]], device=engine.device)
    repeated = [engine.infer(centers, latent) for _ in range(2)]
    repeat_pairs, repeat_valid = resampling_changes(np.array(repeated))
    changes, valid = resampling_changes(np.array(center_outputs))
    rows = []
    for i, sample in enumerate(samples):
        row = table.loc[centers[i]]
        output = fixed[i]
        good = bool(np.isfinite(output).all() and (np.linalg.norm(output[:, :2], axis=1) > EPS).all())
        half = fd_gradient(output[5:9], FD_STEP_M / 2) if good else np.full(2, np.nan)
        coarse = fd_gradient(output[1:5], FD_STEP_M) if good else np.full(2, np.nan)
        gradient = output[0, 2:] / SCALE
        pairs = changes[:, i]
        finite = pairs[np.isfinite(pairs)]
        rows.append({"sample": sample, "x_world_m": row.x_world, "y_world_m": row.y_world,
                     "fixed_valid": good, "generated_gx_rad_per_m": gradient[0], "generated_gy_rad_per_m": gradient[1],
                     "fd_gx_rad_per_m": half[0], "fd_gy_rad_per_m": half[1],
                     "fixed_consistency_rad_per_m": np.linalg.norm(half - gradient),
                     "coarse_consistency_rad_per_m": np.linalg.norm(coarse - gradient),
                     "fd_step_change_rad_per_m": np.linalg.norm(half - coarse),
                     "repeated_render_angle_deg": np.degrees(abs(wrap(angle(output[9]) - angle(output[0])))) if good else np.nan,
                     "fixed_same_image_repeat_deg": repeat_pairs[0, i],
                     "invalid_resampled_predictions": int((~valid[:, i]).sum()),
                     "resampled_pair_count": len(finite),
                     "resampled_pair_mean_deg": finite.mean() if len(finite) else np.nan,
                     "resampled_pair_max_deg": finite.max() if len(finite) else np.nan,
                     **{f"heading_seed_{seed}_deg": float(np.degrees(angle(center_outputs[j][i])))
                        if valid[j, i] else np.nan for j, seed in enumerate(LATENT_SEEDS)}})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "per_point.csv", index=False)
    notes = [f"Images: {DATA}; model: {MODELS['gvc']}",
             f"Positions={POINT_COUNT}; selected samples={samples}.",
             f"Fixed seed={FIXED_SEED}; g0={latents[0]}; shared across all neighborhoods.",
             f"Resampling draws={len(LATENT_SEEDS)} per center image; all unordered pairs evaluated.", "",
             "Fixed-latent consistency (rad/m): " + mean_max(frame.fixed_consistency_rad_per_m),
             "Same-image resampling change (deg): " + mean_max(changes.ravel()),
             "Same-image fixed-latent repeat (deg): " + mean_max(repeat_pairs.ravel()),
             "Coarse-stencil consistency (rad/m): " + mean_max(frame.coarse_consistency_rad_per_m),
             "Gradient step-size change (rad/m): " + mean_max(frame.fd_step_change_rad_per_m),
             "Repeated-render change (deg): " + mean_max(frame.repeated_render_angle_deg),
             f"Invalid neighborhoods={int((~frame.fixed_valid).sum())}; invalid resampled outputs={int((~valid).sum())}; invalid repeated outputs={int((~repeat_valid).sum())}.", "",
             "Circular finite differences use 0.5 m and 1 m steps; gradients are converted to rad/m.",
             "Spatial residuals include rendering and finite-difference effects. Same-image resampling measures latent sensitivity.",
             "The metrics have different units and cannot be subtracted. Sampled checks do not certify uniform bounds.", "",
             "Latent vectors (seed: g0):", *[f"{seed}: {value}" for seed, value in zip(LATENT_SEEDS, latents)],
             "Prediction sources:", *sources]
    (OUT / "summary.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(f"Saved: {OUT / 'per_point.csv'}\nSummary: {OUT / 'summary.txt'}")


if __name__ == "__main__":
    main()
