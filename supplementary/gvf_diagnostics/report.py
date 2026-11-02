import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from supplementary.gvf_diagnostics.settings import (
    COLORS, CODE_KP, DT, JSBSIM_DIR, LABELS, OUT, REFERENCE_KP, REPORTS, SPEED,
    TAU, save_json, sha256, summarize, wrap,
)


def save_figure(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(REPORTS / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_table(frame, name, title):
    frame.to_csv(REPORTS / f"{name}.csv", index=False)
    fig, ax = plt.subplots(figsize=(15, max(3.2, 0.4 * (len(frame) + 2))))
    ax.axis("off")
    display = frame.map(lambda v: "--" if pd.isna(v) else f"{v:.4g}" if isinstance(v, float) else v)
    table = ax.table(cellText=display.to_numpy(), colLabels=frame.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    table.auto_set_column_width(range(len(frame.columns)))
    ax.set_title(title, pad=25)
    save_figure(fig, name)


def jsbsim_series(poses, commands, cfg, vned=None):
    n = min(len(poses), len(commands))
    if vned is not None:
        n = min(n, len(vned))
    if n <= 501:
        raise ValueError("Insufficient aligned log samples for [500:2000]")
    dt, speed = float(cfg.get("control_dt", DT)), float(cfg.get("target_speed", SPEED))
    if dt <= 0:
        raise ValueError("control_dt must be positive")
    ids = np.arange(500, min(2000, n - 1))
    heading = poses[ids, 2]
    actual_rate = wrap(poses[ids + 1, 2] - heading) / dt
# Use the unwrapped ENU command, as in the planar model.
    error = commands[ids] - heading
    model_rate = error / TAU
    target_velocity = speed * np.column_stack((np.cos(heading), np.sin(heading)))
    step = poses[ids + 1, :2] - poses[ids, :2]
    velocity = vned[ids][:, [1, 0]] if vned is not None else step / dt
    table = pd.DataFrame({"frame_index": ids, "time_s": ids * dt,
                          "actual_heading_rate_rad_s": actual_rate,
                          "simplified_heading_rate_rad_s": model_rate,
                          "heading_model_residual_rad_s": abs(actual_rate - model_rate),
                          "velocity_model_residual_m_s": np.linalg.norm(velocity - target_velocity, axis=1),
                          "position_step_residual_m": np.linalg.norm(step - dt * target_velocity, axis=1)})
    if "heading_lookahead" in cfg:
        table["applied_rate_residual_rad_s"] = abs(actual_rate - error / float(cfg["heading_lookahead"]))
    return table


def jsbsim_logs():
    rows, status, sources = [], [], {}
    for folder in sorted(JSBSIM_DIR.glob("*_none")):
        paths = [folder / name for name in ("poses.npy", "heading_commands.npy")]
        record = {"method": folder.name, "source_folder": str(folder)}
        if not all(p.is_file() for p in paths):
            status.append({**record, "status": "missing poses/heading commands"})
            continue
        cfg_path, velocity_path = folder / "run_config.json", folder / "vned_data.npy"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        paths += [p for p in (cfg_path, velocity_path) if p.exists()]
        sources[folder.name] = {p.name: sha256(p) for p in paths}
        try:
            poses, commands = (np.load(p, allow_pickle=False) for p in paths[:2])
            velocity = np.load(velocity_path, allow_pickle=False) if velocity_path.exists() else None
            series = jsbsim_series(poses, commands, cfg, velocity)
        except (ValueError, IndexError) as exc:
            status.append({**record, "status": str(exc)})
            continue
        target = REPORTS / "jsbsim"
        target.mkdir(exist_ok=True)
        series.to_csv(target / f"{folder.name}_residuals.csv", index=False)
        columns = [c for c in series if "residual" in c]
        for metric in columns:
            rows.append({"method": folder.name, "metric": metric, **summarize(
                series.assign(method=folder.name), [metric]).iloc[0].to_dict()})
        status.append({**record, "status": "analyzed", "samples": len(series),
                       "dt_s": cfg.get("control_dt", DT),
                       "dt_source": "saved config" if "control_dt" in cfg else "assumed experiment-3 dt=0.05",
                       "speed_source": "saved config" if "target_speed" in cfg else "assumed experiment-3 speed=23",
                       "velocity_source": "NED velocity log" if velocity is not None else "position finite difference",
                       "applied_rate_available": "heading_lookahead" in cfg})
    pd.DataFrame(status).to_csv(REPORTS / "jsbsim_log_status.csv", index=False)
    save_json(REPORTS / "jsbsim_sources.json", sources)
    if rows:
        save_table(pd.DataFrame(rows), "table3_jsbsim_model_mismatch",
                   "JSBSim vs planar reference model (absolute residuals; [500:2000])")
    else:
        print("No sufficient JSBSim logs found; see jsbsim_log_status.csv.")


def offline_reports():
    path = OUT / "direction_summary.csv"
    if not path.exists():
        print("Offline diagnostics absent; run evaluate.py for Tables 1-2.")
        return
    direction = pd.read_csv(path)
    selection = direction[direction.metric == "angular_error_deg"].copy()
    selection["method"] = selection.method.map(LABELS)
    save_table(selection[["method", "n", "invalid_samples", "Mean", "P95", "Max"]],
               "table1_offgrid_direction_deg", "Off-grid GVF angular error (degrees)")
    local = pd.read_csv(OUT / "local_summary.csv")
    local["method"] = local.method.map(LABELS)
    save_table(local, "table2_local_consistency", "Local gradient, consistency and angular-rate diagnostics")
    for name in ("direction_summary", "direction_by_latent", "latent_jump_summary", "theory_checks", "ideal_critical_points"):
        pd.read_csv(OUT / f"{name}.csv").to_csv(REPORTS / f"{name}.csv", index=False)
    errors = pd.read_csv(OUT / "direction_errors.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    for method, group in errors.groupby("method", sort=False):
        values = np.sort(group.angular_error_deg.to_numpy())
        ax.plot(values, np.arange(1, len(values) + 1) / len(values), color=COLORS[method],
                label=LABELS[method], linewidth=2)
    ax.set(xlabel="Angular error (deg)", ylabel="Empirical cumulative probability")
    ax.grid(True)
    ax.legend()
    save_figure(fig, "offgrid_direction_error_cdf")


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12})
    offline_reports()
    jsbsim_logs()
    notes = f"""Off-grid diagnostics use 200 positions and 60 local neighborhoods in the same environment.
GVC uses five fixed latent draws; localization is evaluated once.
Invalid directions receive 180 deg angular error and Delta=2; invalid gradients are counted as nonfinite.
Delta is the normalized direction difference; gradient errors and consistency are in rad/m.
Circular finite differences use 1 m and 0.5 m steps with a shared latent.
Delta_psi=V*consistency is a sampled worst-heading rate mismatch at V={SPEED} m/s.
The latent-jump proxy is the maximum sampled direction change divided by {DT} s.
Sample inequalities: Delta < 1 and Delta^2+(Delta_psi/kp)^2 < 1.
Checks use implemented kp={CODE_KP} and reference kp={REFERENCE_KP} separately.
Sampled maxima do not certify uniform bounds or the full assumptions of Theorem 2.
JSBSim residuals compare logged yaw rates with (command-yaw)/{TAU} s.
Saved heading_lookahead adds an applied-rate comparison; assumed dt/speed are recorded in jsbsim_log_status.csv.
"""
    (REPORTS / "interpretation.txt").write_text(notes, encoding="utf-8")
    print(f"Reports: {REPORTS}")


if __name__ == "__main__":
    main()
