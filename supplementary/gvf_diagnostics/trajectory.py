import ast
import json

import numpy as np
import pandas as pd

from supplementary.gvf_diagnostics.settings import EPS, ORIGIN, ROOT, SCALE, angle, field, guard, sha256, wrap

SOURCE = ROOT / "data/visual_robustness/run01/gvc/clean/rep_00"
OUT = ROOT / "data/gvf_diagnostics/trajectory/run01"
START, END = 500, 2000


def controller_constants():
    tree = ast.parse((ROOT / "src/generative_visual_control.py").read_text(encoding="utf-8"))
    functions = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    fn = functions["calculate_heading_cmd"]
    defaults = dict(zip([a.arg for a in fn.args.args][-len(fn.args.defaults):], fn.args.defaults))
    kp = float(ast.literal_eval(defaults["kp"]))
    loop_dt = next(ast.literal_eval(n.value) for n in ast.walk(functions["control_loop"])
                   if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "simulation_dt" for t in n.targets))
    return kp, float(loop_dt)


def build_frame_table(poses, predictions, commands, cfg, kp, dt, terms=None):
    n = len(poses)
    if n < 2 or poses.shape != (n, 3) or predictions.shape != (n, 4) or commands.shape != (n,):
        raise ValueError("Positions, predictions and heading commands must be complete and aligned")
    if not all(np.isfinite(a).all() for a in (poses, commands)):
        raise ValueError("Nonfinite true state or heading command")
    speed, tau = float(cfg["v0"]), float(cfg["tau_theta"])
    if tau <= 0 or speed <= 0 or not np.isclose(float(cfg["dt"]), dt):
        raise ValueError("Invalid dynamics parameters or loop/model time-step mismatch")
    truth = np.array([field(p) for p in (poses[:, :2] - ORIGIN) / SCALE])
    raw_norm, true_norm = (np.linalg.norm(a[:, :2], axis=1) for a in (predictions, truth))
    generated_valid = np.isfinite(predictions).all(axis=1) & (raw_norm > EPS)
    ideal_valid = np.isfinite(truth).all(axis=1) & (true_norm > EPS)
    valid = generated_valid & ideal_valid
    pred_unit = np.divide(predictions[:, :2], raw_norm[:, None],
                          out=np.full((n, 2), np.nan), where=generated_valid[:, None])
    true_unit = np.divide(truth[:, :2], true_norm[:, None],
                          out=np.full((n, 2), np.nan), where=ideal_valid[:, None])
    desired = angle(predictions)
    desired[~generated_valid] = np.nan
    heading_error = wrap(desired - poses[:, 2])
    velocity = speed * np.column_stack((np.cos(poses[:, 2]), np.sin(poses[:, 2])))
    raw_ff = np.sum(predictions[:, 2:] * velocity, axis=1) / SCALE
    applied_ff = float(cfg.get("gradient_gain", 1.0)) * raw_ff
    if cfg.get("gradient_clip") is not None:
        applied_ff = np.clip(applied_ff, -float(cfg["gradient_clip"]), float(cfg["gradient_clip"]))
    if terms is not None and (terms.shape != (n, 2) or not np.allclose(
            terms, np.column_stack((raw_ff, applied_ff)), atol=1e-7, rtol=1e-7, equal_nan=True)):
        raise ValueError("Saved gradient terms disagree with generated outputs and units")
    command_replay = poses[:, 2] + tau * (kp * np.sin(heading_error) + applied_ff)
    if not np.allclose(command_replay, commands, atol=1e-6, rtol=1e-7):
        raise ValueError("Control-law replay failed; do not assume the current kp matches these logs")
    step_error = np.linalg.norm(np.diff(poses[:, :2], axis=0) - dt * velocity[:-1], axis=1)
    actual_yaw_rate = np.diff(poses[:, 2]) / dt
    if step_error.max() > 1e-6 or not np.allclose(actual_yaw_rate,
            (commands[:-1] - poses[:-1, 2]) / tau, atol=1e-7, rtol=1e-7):
        raise ValueError("Logs do not follow the stated planar Euler dynamics/time alignment")
    generated_rate = np.r_[wrap(np.diff(desired)) / dt, np.nan]
    epsilon = generated_rate - applied_ff
    yaw_rate = np.r_[actual_yaw_rate, np.nan]
    delta = np.linalg.norm(pred_unit - true_unit, axis=1)
    gradient_error = np.linalg.norm(predictions[:, 2:] - truth[:, 2:], axis=1) / SCALE
    gradient_error[~valid] = np.nan
    signed_angle = wrap(desired - angle(truth))
    signed_angle[~valid] = np.nan
    frame = pd.DataFrame({"frame_index": np.arange(n), "frame_id": np.arange(1, n + 1),
                          "time_s": np.arange(n) * dt, "x_world_m": poses[:, 0], "y_world_m": poses[:, 1],
                          "heading_rad": poses[:, 2], "vx_world_m_s": velocity[:, 0], "vy_world_m_s": velocity[:, 1],
                          "speed_m_s": speed, "generated_raw_norm": raw_norm,
                          "generated_valid": generated_valid, "ideal_valid": ideal_valid,
                          "generated_heading_rad": desired, "heading_error_rad": heading_error,
                          "heading_command_rad": commands, "direction_delta": delta,
                          "direction_angle_error_deg": np.degrees(abs(signed_angle)),
                          "gradient_error_rad_per_m": gradient_error,
                          "generated_rate_fd_rad_s": generated_rate, "raw_gradient_rate_rad_s": raw_ff,
                          "applied_gradient_rate_rad_s": applied_ff,
                          "epsilon_psi_rad_s": epsilon, "consistency_error_rad_s": abs(epsilon),
                          "control_law_replay_residual_rad": command_replay - commands,
                          "heading_dynamics_residual_rad_s": yaw_rate - (commands - poses[:, 2]) / tau,
                          "heading_error_equation_residual_rad_s": generated_rate - yaw_rate + kp * np.sin(heading_error) - epsilon,
                          "interval_dt_s": np.r_[np.full(n - 1, dt), np.nan]})
    for j, axis in enumerate(("x", "y")):
        frame[f"generated_raw_u{axis}"] = predictions[:, j]
        frame[f"generated_unit_u{axis}"] = pred_unit[:, j]
        frame[f"analytic_u{axis}"] = truth[:, j]
        frame[f"generated_g{axis}_rad_per_m"] = predictions[:, j + 2] / SCALE
        frame[f"analytic_g{axis}_rad_per_m"] = truth[:, j + 2] / SCALE
    return frame


def mean_max(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return f"Mean={values.mean():.8g}, Max={values.max():.8g}, n={len(values)}" if len(values) else "unavailable, n=0"


def condition(frame, kp):
    delta = float(frame.direction_delta.max())
    rate = float(frame.consistency_error_rad_s.max())
    valid = bool(frame.generated_valid.all() and frame.ideal_valid.all()
                 and np.isfinite(frame.consistency_error_rad_s).any())
    numeric = bool(np.isfinite(delta + rate) and delta < 1 and rate < kp
                   and delta ** 2 + (rate / kp) ** 2 < 1)
    text = (f"Delta_max={delta:.8g}, Delta_psi_FD_max={rate:.8g} rad/s; "
            f"Delta^2+(Delta_psi/kp)^2={delta ** 2 + (rate / kp) ** 2:.8g}; "
            f"sample numerical inequalities={'PASS' if numeric else 'NOT SATISFIED'}; "
            f"all sampled directions regular={valid}.")
    if not valid:
        text += " Full-window sufficient assumptions are not established, regardless of the numerical inequality result."
    if numeric:
        bar = (np.arcsin(rate / kp) + np.arccos(delta)) / 2
        initial = abs(float(frame.heading_error_rad.iloc[0])) < np.pi - bar
        text += (f" Admissible bar_psi={np.degrees(bar):.6g} deg; "
                 f"heading initial condition at this window start={initial}.")
    return text


def main():
    paths = {name: SOURCE / name for name in (
        "poses.npy", "predicted_vectors.npy", "heading_commands.npy", "run_config.json", "trial_status.json")}
    if any(not p.is_file() for p in paths.values()):
        raise SystemExit(f"Insufficient recorded data: {SOURCE}; set SOURCE to a complete Clean GVC run.")
    cfg = json.loads(paths["run_config.json"].read_text(encoding="utf-8"))
    status = json.loads(paths["trial_status.json"].read_text(encoding="utf-8"))
    if cfg.get("path_type") != "cassini" or cfg.get("visual_condition") != "clean" or status["status"] != "completed":
        raise ValueError("SOURCE must be one completed Clean Cassini GVC trajectory")
    if cfg.get("mode_persistence", False):
        raise ValueError("This diagnostic expects the original per-frame latent resampling")
    terms_path = SOURCE / "gradient_terms.npy"
    if terms_path.exists():
        paths["gradient_terms.npy"] = terms_path
    saved_protocol = SOURCE.parents[2] / "protocol.json"
    if saved_protocol.is_file():
        paths["original_protocol.json"] = saved_protocol
    kp, dt = controller_constants()
    poses, predictions, commands = (np.load(paths[name]) for name in (
        "poses.npy", "predicted_vectors.npy", "heading_commands.npy"))
    if len(poses) < END or int(status["frames"]) != len(poses):
        raise ValueError("Incomplete trajectory or insufficient frames for [500:2000]")
    terms = np.load(terms_path) if terms_path.exists() else None
    frame = build_frame_table(poses, predictions, commands, cfg, kp, dt, terms)
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {"source": str(SOURCE), "inputs": {name: sha256(path) for name, path in paths.items()},
                "code": {name: sha256(ROOT / name) for name in (
                    "src/generative_visual_control.py", "src/fixed_wing_model/fixed_wing.py",
                    "src/path_following/create_path_cassini_oval.py", "supplementary/gvf_diagnostics/trajectory.py")},
                "kp": kp, "tau_s": cfg["tau_theta"], "dt_s": dt, "settled_window": [START, END]}
    guard(OUT / "source_protocol.json", protocol)
    frame.to_csv(OUT / "trajectory_errors.csv", index=False)
    notes = [f"One Clean GVC trajectory: {SOURCE}", f"Frames={len(frame)}, dt={dt} s; speed={cfg['v0']} m/s; kp={kp}; tau_psi={cfg['tau_theta']} s.",
             "Time is simulation time reconstructed from the verified loop step, not wall-clock render time.",
             "Velocity is reconstructed from the known constant-speed planar model and verified against recorded position increments.",
             f"Latent seed={cfg.get('mode_seed')}; latent is resampled each frame. Original samples are used without smoothing or reseeding.", ""]
    for name, part in (("Full recorded trajectory", frame), ("Common settled window [500:2000]", frame.iloc[START:END])):
        notes += [name,
                  "1. Direction Delta (normalized vector difference): " + mean_max(part.direction_delta),
                  "   Same direction error expressed as angle (deg): " + mean_max(part.direction_angle_error_deg),
                  "2. Generated minus analytical orientation-gradient norm (rad/m): " + mean_max(part.gradient_error_rad_per_m),
                  "3. Absolute angular-rate consistency error |epsilon_psi_FD| (rad/s): " + mean_max(part.consistency_error_rad_s),
                  f"   Undefined ideal directions={int((~part.ideal_valid).sum())}; invalid generated outputs={int((~part.generated_valid).sum())}; no-next-frame rates={int(part.interval_dt_s.isna().sum())}.",
                  "Theorem 2, implemented gain: " + condition(part, kp), ""]
    notes += ["Control law: psi_c=psi+tau*(gradient_rate+kp*sin(e)); e_dot=-kp*sin(e)+epsilon_psi.",
              "World gradients are normalized-coordinate gradients divided by 100 (rad/m).",
              "epsilon_psi[k]=wrap(heading_hat[k+1]-heading_hat[k])/dt - applied_gradient_rate[k].",
              "Rates include per-frame latent resampling; the last frame has no forward rate.",
              "Sampled maxima do not certify continuous-time uniform bounds. Failed sufficient checks do not prove instability."]
    (OUT / "summary.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(f"Reused: {SOURCE}\nCSV: {OUT / 'trajectory_errors.csv'}\nSummary: {OUT / 'summary.txt'}")


if __name__ == "__main__":
    main()
