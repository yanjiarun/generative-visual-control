import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from supplementary.gvf_diagnostics.settings import (
    CODE_KP, DATA, DT, EPS, FD_STEP_M, LATENT_SEEDS, MODELS, ODE_STEPS, OUT,
    REFERENCE_KP, SCALE, SPEED, BATCH_SIZE, angle, fd_gradient, field, guard,
    sha256, summarize, wrap,
)


def predict(table, method, seed):
    path = OUT / "predictions" / f"{method}_{seed}.npz"
    if path.exists():
        values = np.load(path)["outputs"]
        if values.shape != (len(table), 4):
            raise ValueError(f"Invalid prediction cache: {path}")
        return values
    from src.baseline.position_localization import CNNPositionRegressor
    from src.generative_model.core.generative_controller import FlowMatchingController, sample_action
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = (FlowMatchingController(history_length=1) if method == "gvc"
             else CNNPositionRegressor()).to(device)
    checkpoint = torch.load(MODELS[method], map_location="cpu")
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval().requires_grad_(False)
    transform = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.CenterCrop((224, 224)), transforms.ToTensor()])
# Share one latent across centers and stencils.
    rng = torch.Generator(device=device).manual_seed(seed)
    latent = torch.rand(1, 4, device=device, generator=rng) * 2 - 1
    values = []
    with torch.inference_mode():
        for start in range(0, len(table), BATCH_SIZE):
            names = table.image.iloc[start:start + BATCH_SIZE]
            batch = torch.stack([transform(Image.fromarray(cv2.cvtColor(
                cv2.imread(str(DATA / "images" / name)), cv2.COLOR_BGR2RGB))) for name in names]).to(device)
            if method == "gvc":
                result = sample_action(model, batch.unsqueeze(1), ODE_STEPS, device,
                                       initial_actions=latent.expand(len(batch), -1)).cpu().numpy()
            else:
                positions = model(batch).cpu().numpy()
                result = np.array([field(p) if np.isfinite(p).all() else np.full(4, np.nan) for p in positions])
            values.extend(result)
            print(f"{method}, latent {seed}: {min(start + len(batch), len(table))}/{len(table)}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, outputs=values, latent=latent.cpu().numpy())
    temporary.replace(path)
    return values


def direction_metrics(predicted, truth):
    norm = np.linalg.norm(predicted[:2])
    valid = bool(np.isfinite(predicted).all() and norm > EPS)
    return {"valid": valid, "raw_direction_norm": norm,
            "angular_error_deg": float(np.degrees(abs(wrap(angle(predicted) - angle(truth))))) if valid else 180.0,
            "delta": float(np.linalg.norm(predicted[:2] / norm - truth[:2] / np.linalg.norm(truth[:2]))) if valid else 2.0,
            "delta_g_rad_per_m": float(np.linalg.norm(predicted[2:] - truth[2:]) / SCALE) if valid else np.nan}


def local_metrics(predictions, truth, indices):
    center = predictions[indices["center"]]
    true = truth[indices["center"]]
    valid = all(np.isfinite(predictions[i]).all() and np.linalg.norm(predictions[i, :2]) > EPS
                for i in indices.values())
    grad = {}
    for tag, step in (("h", FD_STEP_M), ("half", FD_STEP_M / 2)):
        ids = [indices[f"{tag}_{axis}"] for axis in ("xp", "xm", "yp", "ym")]
        grad[tag] = fd_gradient(predictions[ids], step) if valid else np.full(2, np.nan)
    error = grad["half"] - center[2:] / SCALE
    true_ids = [indices[f"half_{axis}"] for axis in ("xp", "xm", "yp", "ym")]
    values = {**direction_metrics(center, true), "local_valid": valid,
              "fd_gx_rad_per_m": grad["half"][0], "fd_gy_rad_per_m": grad["half"][1],
              "consistency_rad_per_m": np.linalg.norm(error),
              "delta_psi_worst_rad_s": SPEED * np.linalg.norm(error),
              "coarse_delta_psi_worst_rad_s": SPEED * np.linalg.norm(grad["h"] - center[2:] / SCALE),
              "fd_step_change_rad_per_m": np.linalg.norm(grad["half"] - grad["h"]),
              "fd_gradient_error_rad_per_m": np.linalg.norm(grad["half"] - true[2:] / SCALE),
              "ideal_fd_check_rad_per_m": np.linalg.norm(fd_gradient(truth[true_ids], FD_STEP_M / 2) - true[2:] / SCALE),
              "repeat_angle_deg": np.degrees(abs(wrap(angle(predictions[indices["repeat_center"]]) - angle(center)))) if valid else np.nan}
    return values


def theory_check(delta, rate, kp, valid):
    """Existence of bar_psi: asin(rate/kp) < bar_psi < acos(delta). Empirical only."""
    feasible = bool(valid and np.isfinite(delta + rate) and delta < 1 and rate < kp
                    and delta ** 2 + (rate / kp) ** 2 < 1)
    result = {"delta_max": delta, "delta_psi_max_rad_s": rate, "kp": kp,
              "all_predictions_valid": bool(valid), "sample_inequalities_satisfied": feasible,
              "uniform_theorem_certified": False,
              "bar_psi_deg": np.nan, "heading_margin_rad_s": np.nan,
              "direction_margin": np.nan, "L_star": np.nan}
    if feasible:
        bar = (np.arcsin(rate / kp) + np.arccos(delta)) / 2
# Solve cos(bar) G(L) + sin(bar) H(L) + delta = 0.
        low, high = 0.0, 1.0
        for _ in range(80):
            mid = (low + high) / 2
            if -np.cos(bar) * mid + np.sin(bar) * np.sqrt(1 - mid ** 2) + delta > 0:
                low = mid
            else:
                high = mid
        result.update(bar_psi_deg=float(np.degrees(bar)), heading_margin_rad_s=float(kp * np.sin(bar) - rate),
                      direction_margin=float(np.cos(bar) - delta),
                      L_star=float((np.tan(np.pi * (low + high) / 4) / 2) ** 2))
    return result


def main():
    manifest = DATA / "manifest.csv"
    if not manifest.exists():
        raise SystemExit("First run: python -m supplementary.gvf_diagnostics.collect")
    table = pd.read_csv(manifest)
    images = [DATA / "images" / name for name in table.image]
    if any(not p.is_file() for p in images):
        raise SystemExit("Image collection incomplete; run collect.py again.")
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {"manifest": sha256(manifest), "collection": sha256(DATA / "protocol.json"),
                "images": {p.name: sha256(p) for p in images},
                "models": {name: sha256(path) for name, path in MODELS.items()},
                "latent_seeds": list(LATENT_SEEDS), "ode_steps": ODE_STEPS,
                "fd_step_m": FD_STEP_M, "speed_m_s": SPEED, "dt": DT,
                "code_kp": CODE_KP, "reference_kp": REFERENCE_KP}
    guard(OUT / "evaluation_protocol.json", protocol)
    truth = np.array([field(p) for p in table[["x_norm", "y_norm"]].to_numpy()])
    centers = table[table.view == "center"]
    direction, local, jumps = [], [], []
    for method in MODELS:
        angles, validity = [], []
        for seed in (LATENT_SEEDS if method == "gvc" else (LATENT_SEEDS[0],)):
            outputs = predict(table, method, seed)
            for row in centers.itertuples():
                direction.append({"method": method, "latent_seed": seed, "sample": row.sample,
                                  **direction_metrics(outputs[row.Index], truth[row.Index])})
            angles.append(angle(outputs[centers.index]))
            validity.append(np.isfinite(outputs[centers.index]).all(axis=1) &
                            (np.linalg.norm(outputs[centers.index, :2], axis=1) > EPS))
            for sample, group in table[table.local].groupby("sample", sort=False):
                indices = dict(zip(group.view, group.index))
                local.append({"method": method, "latent_seed": seed, "sample": sample,
                              **local_metrics(outputs, truth, indices)})
        if method == "gvc":
            a = np.array(angles)
            max_jump = np.max(np.abs(wrap(a[:, None] - a[None, :])), axis=(0, 1))
            for sample, jump, valid in zip(centers["sample"], max_jump, np.all(validity, axis=0)):
                jumps.append({"method": method, "sample": sample, "valid": bool(valid),
                              "max_sampled_latent_jump_deg": float(np.degrees(jump)) if valid else np.nan,
                              "jump_rate_proxy_rad_s": float(jump / DT) if valid else np.nan})
    direction, local, jumps = pd.DataFrame(direction), pd.DataFrame(local), pd.DataFrame(jumps)
    for name, frame in (("direction_errors", direction), ("local_errors", local), ("latent_jumps", jumps)):
        frame.to_csv(OUT / f"{name}.csv", index=False)
    summarize(direction, ["angular_error_deg", "delta", "delta_g_rad_per_m"]).to_csv(OUT / "direction_summary.csv", index=False)
    summarize(direction, ["angular_error_deg", "delta", "delta_g_rad_per_m"],
              ("method", "latent_seed")).to_csv(OUT / "direction_by_latent.csv", index=False)
    metrics = ["delta_g_rad_per_m", "consistency_rad_per_m", "delta_psi_worst_rad_s",
               "coarse_delta_psi_worst_rad_s", "fd_step_change_rad_per_m", "fd_gradient_error_rad_per_m",
               "ideal_fd_check_rad_per_m", "repeat_angle_deg"]
    summarize(local, metrics).to_csv(OUT / "local_summary.csv", index=False)
    summarize(jumps, ["max_sampled_latent_jump_deg", "jump_rate_proxy_rad_s"]).to_csv(OUT / "latent_jump_summary.csv", index=False)
    checks = []
    for method, group in direction.groupby("method", sort=False):
        subset = local[local.method == method]
        valid = bool(group.valid.all() and subset.local_valid.all())
        delta = float(group.delta.max())
        rate = float(subset[["delta_psi_worst_rad_s", "coarse_delta_psi_worst_rad_s"]].max().max())
        scopes = {"fixed_latent_spatial": rate}
        if method == "gvc":
            valid = valid and bool(jumps.valid.all())
            scopes["spatial_plus_sampled_jump_proxy"] = rate + float(jumps.jump_rate_proxy_rad_s.max())
        for scope, bound in scopes.items():
            for label, kp in (("implemented", CODE_KP), ("reference_gain", REFERENCE_KP)):
                checks.append({"method": method, "scope": scope, "gain_source": label,
                               **theory_check(delta, bound, kp, valid)})
    pd.DataFrame(checks).to_csv(OUT / "theory_checks.csv", index=False)

    critical = np.array([[0., 0.], [-1.5, 0.], [1.5, 0.]])
    alpha = np.sum(critical ** 2, axis=1) ** 2 - 4.5 * (critical[:, 0] ** 2 - critical[:, 1] ** 2) - 0.9 ** 4
    pd.DataFrame({"x_norm": critical[:, 0], "y_norm": critical[:, 1], "alpha": alpha,
                  "L": alpha ** 2 / 2, "field_norm": [np.linalg.norm(field(p)[:2]) for p in critical]}).to_csv(
                      OUT / "ideal_critical_points.csv", index=False)
    print(f"Evaluation saved: {OUT}\nNext: python -m supplementary.gvf_diagnostics.report")


if __name__ == "__main__":
    main()
