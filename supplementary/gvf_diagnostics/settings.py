import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.path_following.create_path_cassini_oval import compute_cassini_field_with_gradients

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/gvf_diagnostics/offgrid" / "run01"
DATA = OUT / "observations"
REPORTS = OUT / "reports"
GRID = ROOT / "src/path_following/vector_field_with_gradients_cassini.csv"
MODELS = {
    "gvc": ROOT / "src/generative_model/models/flow_matching_cassini_models/flow_matching_policy_final.pth",
    "localization_gvf": ROOT / "src/baseline/models/position_cassini_cnn/position_best.pth",
}
LABELS = {"gvc": "Proposed GVC", "localization_gvf": "VisLoc-GVF"}
COLORS = {"gvc": "darkred", "localization_gvf": "purple"}
POINTS, LOCAL_POINTS, SEED = 200, 60, 20261027
LATENT_SEEDS = tuple(range(SEED, SEED + 5))
BOUNDS = np.array([[-2.8, 2.8], [-1.4, 1.4]])
ORIGIN, SCALE = np.array([576.0, 275.0]), 100.0
FD_STEP_M, MIN_GRID_DISTANCE_M = 1.0, 1.2

SPEED, DT, TAU, CODE_KP, REFERENCE_KP = 23.0, 0.05, 6.0, 0.18, 1.0
BATCH_SIZE, ODE_STEPS, EPS = 16, 50, 1e-8
JSBSIM_DIR = ROOT / "jsbsim_data/cassini_path_following_data"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def guard(path, protocol):
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != protocol:
        raise ValueError(f"Protocol changed: choose a new OUT directory in settings.py ({path})")
    save_json(path, protocol)


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def field(xy):
    """Return [Phi_x, Phi_y, g_x, g_y]; g uses normalized coordinates, as in control."""
    ux, uy, _, dx, dy = compute_cassini_field_with_gradients(*np.asarray(xy, float))
    return np.r_[ux, uy, ux * dy - uy * dx]


def angle(output):
    return np.arctan2(output[..., 1], output[..., 0])


def fd_gradient(outputs, h_m):
    """Circular central differences; order is x+, x-, y+, y-. Units: rad/m."""
    angles = angle(np.asarray(outputs))
    return wrap(angles[[0, 2]] - angles[[1, 3]]) / (2 * h_m)


def summarize(table, columns, groups=("method",)):
    rows = []
    for keys, group in table.groupby(list(groups), sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        validity = group.get("local_valid", group.get("valid", pd.Series(True, index=group.index)))
        for metric in columns:
            values = group[metric].to_numpy(float)
            valid = values[np.isfinite(values)]
            rows.append({**dict(zip(groups, keys)), "metric": metric, "n": len(values),
                         "finite": len(valid), "nonfinite": len(values) - len(valid),
                         "invalid_samples": int((~validity.astype(bool)).sum()),
                         "Mean": float(valid.mean()) if len(valid) else np.nan,
                         "P95": float(np.percentile(valid, 95)) if len(valid) else np.nan,
                         "Max": float(valid.max()) if len(valid) else np.nan})
    return pd.DataFrame(rows)
