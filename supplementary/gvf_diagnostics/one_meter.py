import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from supplementary.gvf_diagnostics.settings import (
    BOUNDS, DATA, DT, EPS, MODELS, ODE_STEPS, OUT as OFFGRID, ROOT, SCALE, SPEED,
    angle, field, guard, sha256, wrap,
)
from supplementary.gvf_diagnostics.latent_consistency import FIXED_SEED, OUT as PREVIOUS, Predictor, mean_max, select_points

OUT = ROOT / "data/gvf_diagnostics/one_meter/run01"
STEP_M = 1.0


def make_plan(table, samples):
    rows = []
    for sample in samples:
        center = table[table["sample"].eq(sample) & table.view.eq("center")].iloc[0]
        xy = center[["x_norm", "y_norm"]].to_numpy(float)
        direction = field(xy)[:2]
        if not np.isfinite(direction).all() or np.linalg.norm(direction) <= EPS:
            raise ValueError(f"Undefined representative flight direction at sample {sample}")
        delta = STEP_M * direction / np.linalg.norm(direction)
        end = xy + delta / SCALE
        if np.any(end < BOUNDS[:, 0]) or np.any(end > BOUNDS[:, 1]):
            raise ValueError(f"Sample {sample}: 1 m endpoint outside the Cassini region")
        for view, offset in (("start", np.zeros(2)), ("end", delta)):
            world = center[["x_world", "y_world"]].to_numpy(float) + offset
            rows.append({"sample": sample, "view": view, "x_world": world[0], "y_world": world[1],
                         "dx_m": delta[0], "dy_m": delta[1],
                         "image": str(OUT / "images" / f"{sample:03d}_{view}.png")})
    return pd.DataFrame(rows)


def capture(plan):
    camera = None
    try:
        for i, row in enumerate(plan.itertuples()):
            path = Path(row.image)
            if path.exists():
                if cv2.imread(str(path)) is None:
                    raise ValueError(f"Unreadable saved image: {path}")
                continue
            if camera is None:
                from supplementary.visual_robustness.camera import Camera
                camera = Camera()
            image = camera.get_panoramic_image(i + 1, [row.x_world, row.y_world])
            temporary = path.with_suffix(".tmp.png")
            if not cv2.imwrite(str(temporary), image):
                raise OSError(f"Cannot save {path}")
            temporary.replace(path)
            print(f"Capture {i + 1}/{len(plan)}: {row.sample}, {row.view}", flush=True)
    finally:
        if camera is not None:
            camera.close()


def step_error(start, end, delta_m):
    actual = wrap(angle(end) - angle(start))
    predicted = start[2:] @ delta_m / SCALE
    return float(actual), float(predicted), float(abs(wrap(actual - predicted)))


def main():
    if not (DATA / "manifest.csv").exists():
        raise SystemExit("Existing images missing; first run: python -m supplementary.gvf_diagnostics.collect")
    table = pd.read_csv(DATA / "manifest.csv")
    old_path = PREVIOUS / "source_protocol.json"
    previous = json.loads(old_path.read_text(encoding="utf-8")) if old_path.exists() else {}
    samples = previous.get("samples", select_points(table))
    if len(samples) != 30 or len(set(samples)) != 30:
        raise ValueError("Expected the same 30 positions as the previous latent experiment")
    centers = table[table.view.eq("center")].set_index("sample").loc[samples]
    cache_path = OFFGRID / "predictions" / f"gvc_{FIXED_SEED}.npz"
    evaluation_path = OFFGRID / "evaluation_protocol.json"
    if not cache_path.exists() or not evaluation_path.exists():
        raise SystemExit("Fixed-latent cache missing; first run: python -m supplementary.gvf_diagnostics.evaluate")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    model_hash = sha256(MODELS["gvc"])
    if (evaluation["models"]["gvc"] != model_hash or evaluation["manifest"] != sha256(DATA / "manifest.csv")
            or evaluation["ode_steps"] != ODE_STEPS):
        raise ValueError("Existing model/manifest/ODE protocol differs; do not mix experiments")
    if previous and (previous["model_sha256"] != model_hash or previous["latent_seeds"][0] != FIXED_SEED):
        raise ValueError("Previous 30-point experiment used another model or fixed latent seed")
    if FIXED_SEED not in evaluation["latent_seeds"]:
        raise ValueError("Requested g0 seed is absent from the source protocol")
    if previous.get("cache_sha256", {}).get(str(cache_path), sha256(cache_path)) != sha256(cache_path):
        raise ValueError("Original fixed-latent cache changed")
    with np.load(cache_path) as saved:
        latent = saved["latent"].copy()
    if latent.shape != (1, 4) or not np.isfinite(latent).all() or np.any(abs(latent) > 1):
        raise ValueError("Invalid saved g0")
    old_images = [DATA / "images" / name for name in centers.image]
    if any(evaluation["images"].get(p.name) != sha256(p) for p in old_images):
        raise ValueError("Existing center images changed")
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    protocol = {"samples": samples, "step_m": STEP_M, "direction": "ideal analytical GVF at start",
                "fixed_seed": FIXED_SEED, "g0": latent.tolist(), "model_sha256": model_hash,
                "old_centers": {str(p): sha256(p) for p in old_images}, "source_cache": sha256(cache_path),
                "origin_manifest": sha256(DATA / "manifest.csv"), "ode_steps": ODE_STEPS,
                "camera": "experiment-1 world-aligned, 75 m altitude; same frozen model and preprocessing"}
    guard(OUT / "capture_protocol.json", protocol)
    plan = make_plan(table, samples)
    manifest = OUT / "manifest.csv"
    if manifest.exists():
        pd.testing.assert_frame_equal(pd.read_csv(manifest), plan, check_dtype=False, atol=1e-10, rtol=0)
    else:
        plan.to_csv(manifest, index=False)
    capture(plan)
    images = [Path(p) for p in plan.image]
    inputs = {**{str(p): sha256(p) for p in images}, "script": sha256(__file__),
              "previous_csv": sha256(PREVIOUS / "per_point.csv") if (PREVIOUS / "per_point.csv").exists() else None}
    guard(OUT / "evaluation_protocol.json", inputs)

    inference_table = pd.DataFrame({"image": [str(p) for p in old_images] + plan.image.tolist()})
    engine = Predictor(inference_table)
    outputs = engine.infer(list(range(len(inference_table))), torch.as_tensor(latent, device=engine.device))
    old_outputs, paired = outputs[:30], outputs[30:].reshape(30, 2, 4)
    rows = []
    for i, sample in enumerate(samples):
        start, end = paired[i]
        p = plan.iloc[2 * i]
        delta = p[["dx_m", "dy_m"]].to_numpy(float)
        valid = bool(np.isfinite(paired[i]).all() and (np.linalg.norm(paired[i, :, :2], axis=1) > EPS).all())
        actual, predicted, error = step_error(start, end, delta) if valid else (np.nan,) * 3
        render_valid = np.isfinite(old_outputs[i]).all() and np.linalg.norm(old_outputs[i, :2]) > EPS
        render_error = abs(wrap(angle(start) - angle(old_outputs[i]))) if valid and render_valid else np.nan
        rows.append({"sample": sample, "x_world_m": p.x_world, "y_world_m": p.y_world,
                     "end_x_world_m": p.x_world + delta[0], "end_y_world_m": p.y_world + delta[1],
                     "dx_m": delta[0], "dy_m": delta[1], "step_m": np.linalg.norm(delta), "valid": valid,
                     "start_heading_rad": float(angle(start)), "end_heading_rad": float(angle(end)),
                     "start_gx_rad_per_m": start[2] / SCALE, "start_gy_rad_per_m": start[3] / SCALE,
                     "actual_change_rad": actual, "predicted_change_rad": predicted, "error_rad": error,
                     "actual_change_deg": np.degrees(actual), "predicted_change_deg": np.degrees(predicted),
                     "error_deg": np.degrees(error), "zero_prediction_error_deg": np.degrees(abs(actual)),
                     "old_vs_fresh_start_deg": np.degrees(render_error)})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "per_point.csv", index=False)
    improved = int((frame.error_deg < frame.zero_prediction_error_deg).sum())
    notes = [f"One-meter consistency; samples={samples}.",
             f"Fixed seed={FIXED_SEED}; g0={latent.tolist()}.",
             f"Displacement={STEP_M} m along the ideal start direction; control-step distance={SPEED * DT} m.",
             "actual=wrap(heading_end-heading_start); predicted=(gradient_start/100)^T*displacement_m.", "",
             "Consistency error (deg): " + mean_max(frame.error_deg),
             "Consistency error (rad): " + mean_max(frame.error_rad),
             "Absolute actual change (deg): " + mean_max(abs(frame.actual_change_deg)),
             "Absolute predicted change (deg): " + mean_max(abs(frame.predicted_change_deg)),
             "Zero-change reference error (deg): " + mean_max(frame.zero_prediction_error_deg),
             "Old versus fresh start (deg): " + mean_max(frame.old_vs_fresh_start_deg),
             f"Gradient improves on zero-change reference: {improved}/{int(frame.valid.sum())} valid points; invalid pairs={int((~frame.valid).sum())}."]
    previous_csv = PREVIOUS / "per_point.csv"
    if previous_csv.exists():
        old = pd.read_csv(previous_csv).set_index("sample").loc[samples]
        notes += ["", "Previous same-position diagnostics:",
                  "Fixed-latent spatial residual (rad/m): " + mean_max(old.fixed_consistency_rad_per_m),
                  "Resampling: mean pair angle at each point (deg): " + mean_max(old.resampled_pair_mean_deg),
                  "Resampling: largest pair angle at each point (deg): " + mean_max(old.resampled_pair_max_deg)]
    notes += ["", "Residuals include rendering, gradient approximation and finite-step effects.",
              "Fixed latents exclude switching within each pair. Sampled checks do not certify uniform bounds."]
    (OUT / "summary.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(f"CSV: {OUT / 'per_point.csv'}\nSummary: {OUT / 'summary.txt'}")


if __name__ == "__main__":
    main()
