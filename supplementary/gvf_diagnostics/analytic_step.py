import json

import numpy as np
import pandas as pd

from supplementary.gvf_diagnostics.settings import EPS, ORIGIN, ROOT, SCALE, guard, sha256, wrap
from src.path_following.create_path_cassini_oval import compute_cassini_field

SOURCE = ROOT / "data/gvf_diagnostics/one_meter/run01"
OUT = ROOT / "data/gvf_diagnostics/analytic_step/run01"


def compare(frame):
    result = frame.copy()
    if len(frame) != 30 or frame["sample"].duplicated().any():
        raise ValueError("Expected the same 30 unique source samples")
    start = frame[["x_world_m", "y_world_m"]].to_numpy(float)
    end = frame[["end_x_world_m", "end_y_world_m"]].to_numpy(float)
    delta = frame[["dx_m", "dy_m"]].to_numpy(float)
    if (not np.isfinite(np.r_[start, end, delta]).all()
            or not np.allclose(end - start, delta, atol=1e-8, rtol=0)
            or not np.allclose(np.linalg.norm(delta, axis=1), 1.0, atol=1e-8, rtol=0)):
        raise ValueError("Source endpoints/displacements do not describe the saved 1 m steps")
    flags = frame.valid.astype(str).str.lower().map({"true": True, "false": False})
    if flags.isna().any():
        raise ValueError("Unrecognized source validity flag")
    valid = flags.to_numpy(bool)
    result["valid"] = valid
    gradient = frame[["start_gx_rad_per_m", "start_gy_rad_per_m"]].to_numpy(float)
    predicted = np.sum(gradient * delta, axis=1)  # Saved gradients are in rad/m.
    generated = wrap(frame.end_heading_rad.to_numpy() - frame.start_heading_rad.to_numpy())
    source_error = abs(wrap(generated - predicted))
    for actual, saved in ((predicted, frame.predicted_change_rad),
                          (wrap(generated - frame.actual_change_rad), np.zeros(len(frame))),
                          (source_error, frame.error_rad)):
        if not np.allclose(np.asarray(actual)[valid], np.asarray(saved)[valid], atol=1e-7, rtol=2e-6):
            raise ValueError("Saved gradient, directions and original 1 m errors do not agree")
    headings, regular = [], []
    for positions in (start, end):
        q = (positions - ORIGIN) / SCALE
        ux, uy, _ = compute_cassini_field(q[:, 0], q[:, 1])
        headings.append(np.arctan2(uy, ux))
        regular.append(np.isfinite(ux + uy) & (np.hypot(ux, uy) > EPS))
    analytic_valid = regular[0] & regular[1]
    gvf_change = wrap(headings[1] - headings[0])
    paired = valid & analytic_valid & np.isfinite(predicted + generated)
    analytic_residual = wrap(gvf_change - predicted)
    generated_residual = wrap(generated - predicted)
# r_GVC = wrap(r_GVF + change_mismatch).
    mismatch = wrap(generated - gvf_change)
    for name, values in {"gvf_start_heading_rad": headings[0], "gvf_end_heading_rad": headings[1],
                         "gvf_change_rad": gvf_change, "analytic_signed_residual_rad": analytic_residual,
                         "gvc_signed_residual_rad": generated_residual,
                         "gvc_minus_gvf_change_rad": mismatch}.items():
        result[name] = np.where(paired, values, np.nan)
    result["analytic_valid"], result["paired_valid"] = analytic_valid, paired
    result["analytic_vs_gradient_error_rad"] = abs(result.analytic_signed_residual_rad)
    result["gvc_vs_gradient_error_rad"] = abs(result.gvc_signed_residual_rad)
    for name in ("gvf_change", "analytic_vs_gradient_error", "gvc_vs_gradient_error", "gvc_minus_gvf_change"):
        result[f"{name}_deg"] = np.degrees(result[f"{name}_rad"])
    result["analytic_zero_change_error_deg"] = abs(result.gvf_change_deg)
    return result


def mean_max(values):
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    return f"Mean={finite.mean():.8g}, Max={finite.max():.8g} (n={len(finite)})" if len(finite) else "unavailable (n=0)"


def main():
    path = SOURCE / "per_point.csv"
    if not path.is_file():
        raise SystemExit("First complete the previous experiment: python -m supplementary.gvf_diagnostics.one_meter")
    frame = pd.read_csv(path)
    protocol_path = SOURCE / "capture_protocol.json"
    source_protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.exists() else {}
    if source_protocol and frame["sample"].tolist() != source_protocol["samples"]:
        raise ValueError("Sample IDs/order differ from the original capture protocol")
    result = compare(frame)
    OUT.mkdir(parents=True, exist_ok=True)
    guard(OUT / "source_protocol.json", {"source_csv": str(path), "sha256": sha256(path),
          "capture_protocol_sha256": sha256(protocol_path) if protocol_path.exists() else None,
          "script_sha256": sha256(__file__), "origin_m": ORIGIN.tolist(), "scale_m": SCALE,
          "analytic_code_sha256": sha256(ROOT / "src/path_following/create_path_cassini_oval.py")})
    result.to_csv(OUT / "per_point.csv", index=False)
    paired = result[result.paired_valid]
    a, b = paired.analytic_vs_gradient_error_deg, paired.gvc_vs_gradient_error_deg
    closer = int((a < b).sum())
    improved = int((a < paired.analytic_zero_change_error_deg).sum())
    notes = [f"Source: {path}; same 30 one-meter displacements and saved gradients.",
             f"Fixed g0: {source_protocol.get('g0', 'see original summary')}.",
             "Error=abs(wrap(direction_change - gradient^T * displacement_m)); gradients are in rad/m.", "",
             "Analytical GVF error (deg): " + mean_max(a),
             "GVC direction-change error (deg): " + mean_max(b),
             "Analytical GVF error (rad): " + mean_max(paired.analytic_vs_gradient_error_rad),
             "GVC direction-change error (rad): " + mean_max(paired.gvc_vs_gradient_error_rad),
             "Zero-change reference error (deg): " + mean_max(paired.analytic_zero_change_error_deg),
             "Absolute GVC-minus-GVF change (deg): " + mean_max(abs(paired.gvc_minus_gvf_change_deg)),
             f"Analytical error below GVC error: {closer}/{len(paired)} points.",
             f"Gradient improves on zero-change reference: {improved}/{len(paired)} points.",
             f"Invalid source pairs={int((~result.valid).sum())}; undefined analytical pairs={int((~result.analytic_valid).sum())}; valid pairs={len(paired)}/30.", "",
             "Residuals include gradient approximation and finite-step effects. These sampled checks do not certify uniform bounds."]
    (OUT / "summary.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(f"CSV: {OUT / 'per_point.csv'}\nSummary: {OUT / 'summary.txt'}")


if __name__ == "__main__":
    main()
