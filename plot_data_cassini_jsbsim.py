import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_data_cassini import (
    build_cassini_zero_contour,
    calculate_algebraic_path_errors,
    calculate_geometric_path_errors,
    save_localization_statistics,
    _error_summary,
)
from src.path_following.create_path_cassini_oval import compute_cassini_field


DT = 0.05
BASE_X, BASE_Y, SCALE = 576.0, 275.0, 100.0
COLORS = ["gray", "darkblue", "darkorange", "darkred", "purple"]


def load_jsbsim_data(folder, max_length=None):
    """Load and align the arrays saved by VisualControlJSBSim."""
    names = {
        "positions": "positions.npy",
        "vah_commands": "vah_commands.npy",
        "vned": "vned_data.npy",
        "airspeed": "vair_data.npy",
        "attitude": "atti_data.npy",
        "actuators": "commands.npy",
    }
    data = {}
    for key, name in names.items():
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing {path}")
        data[key] = np.asarray(np.load(path, allow_pickle=True))

    lengths = [len(values) for values in data.values()]
    n = min(lengths)
    if max_length is not None:
        n = min(n, int(max_length))
    for key in data:
        data[key] = data[key][:n]
    if data["actuators"].ndim == 1:
        data["actuators"] = data["actuators"][:, None]
    return data


def wrap_to_pi(values):
    values = np.asarray(values, dtype=float)
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def _save_figure(fig, path):
    fig.savefig(f"{path}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{path}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_jsbsim_diagnostics(data, label, output_folder, dt=DT):
    """Save the five dynamics-specific plots for one controller."""
    method_folder = os.path.join(output_folder, label.lower().replace(" ", "_"))
    os.makedirs(method_folder, exist_ok=True)
    time = np.arange(len(data["positions"])) * dt
    positions = data["positions"]
    vah = data["vah_commands"]
    attitude = data["attitude"]
    actuators = data["actuators"]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time, positions[:, 2], color="darkblue", label="Actual altitude", linewidth=2)
    ax.plot(time, vah[:, 2], color="darkorange", linestyle="--",
            label="Altitude command", linewidth=2)
    ax.set(xlabel="Time (s)", ylabel="Altitude (m)", title=f"{label}: altitude")
    ax.set_ylim(70.0, 80.0)
    ax.legend()
    ax.grid(True)
    _save_figure(fig, os.path.join(method_folder, "altitude_command_vs_actual"))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time, data["airspeed"], color="darkblue", label="Airspeed", linewidth=2)
    ax.plot(time, vah[:, 0], color="darkorange", linestyle="--",
            label="Speed command", linewidth=2)
    ax.set(xlabel="Time (s)", ylabel="Speed (m/s)", title=f"{label}: airspeed")
    ax.set_ylim(0.0, 30.0)
    ax.legend()
    ax.grid(True)
    _save_figure(fig, os.path.join(method_folder, "speed_command_vs_airspeed"))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time, attitude[:, 0], color="darkblue", label="Roll", linewidth=2)
    ax.plot(time, attitude[:, 1], color="darkorange", label="Pitch", linewidth=2)
    ax.set(xlabel="Time (s)", ylabel="Attitude (rad)", title=f"{label}: attitude")
    ax.set_ylim(-np.pi / 2.0, np.pi / 2.0)
    ax.legend()
    ax.grid(True)
    _save_figure(fig, os.path.join(method_folder, "roll_pitch_attitude"))

    command_heading = wrap_to_pi(np.deg2rad(vah[:, 1]))
    actual_heading = wrap_to_pi(attitude[:, 2])
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time, actual_heading, color="darkblue", label="Actual heading", linewidth=2)
    ax.plot(time, command_heading, color="darkorange", linestyle="--",
            label="Heading command", linewidth=2)
    ax.set(xlabel="Time (s)", ylabel="Heading (rad)", title=f"{label}: heading")
    ax.set_ylim(-np.pi, np.pi)
    ax.set_yticks([-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0, np.pi])
    ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    ax.legend()
    ax.grid(True)
    _save_figure(fig, os.path.join(method_folder, "heading_command_vs_actual"))

    channel_names = ("Throttle", "Aileron", "Elevator", "Rudder")
    fig, ax = plt.subplots(figsize=(12, 6))
    for index, name in enumerate(channel_names[:actuators.shape[1]]):
        ax.plot(time, actuators[:, index], label=name, linewidth=2)
    ax.set(xlabel="Time (s)", ylabel="Normalized command",
           title=f"{label}: throttle and control surfaces")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(ncol=2)
    ax.grid(True)
    _save_figure(fig, os.path.join(method_folder, "actuator_commands"))


def _plot_reference_curve(ax):
    x = np.linspace(-2.8, 2.8, 300)
    y = np.linspace(-1.4, 1.4, 160)
    x, y = np.meshgrid(x, y)
    _, _, alpha = compute_cassini_field(x, y)
    ax.contour(BASE_X + SCALE * x, BASE_Y + SCALE * y, alpha,
               levels=[0], colors="darkgreen", linewidths=5, linestyles="--")


def compare_jsbsim_data(data_folders, labels, output_folder,
                        max_length=2000, dt=DT):
    """Create the experiment-1 comparison plots and JSBSim diagnostics."""
    data = [load_jsbsim_data(folder, max_length) for folder in data_folders]
    os.makedirs(output_folder, exist_ok=True)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 24

    fig, ax = plt.subplots(figsize=(12, 10))
    for values, label, color in zip(data, labels, COLORS):
        ax.plot(values["positions"][:, 0], values["positions"][:, 1],
                color=color, label=label, linewidth=3, alpha=0.8)
    _plot_reference_curve(ax)
    ax.plot([], [], color="darkgreen", linestyle="--", linewidth=5, label="Desired Curve")
    ax.set(xlabel="East (m)", ylabel="North (m)")
    ax.set_xlim(330, 820)
    ax.set_ylim(90, 490)
    ax.axis("scaled")
    ax.legend()
    ax.grid(True)
    _save_figure(fig, os.path.join(output_folder, "trajectory_comparison_cassini"))

    contour = build_cassini_zero_contour()
    algebraic_stats, geometric_stats = [], []
    algebraic_series, geometric_series = [], []
    fig_alg, ax_alg = plt.subplots(figsize=(12, 8))
    fig_geo, ax_geo = plt.subplots(figsize=(12, 8))
    for values, label, color in zip(data, labels, COLORS):
        algebraic = calculate_algebraic_path_errors(values["positions"])
        geometric = calculate_geometric_path_errors(
            values["positions"], contour_segments=contour)
        mean, maximum, rmse, std = _error_summary(algebraic)
        algebraic_stats.append({"Method": label, "Mean_Error": mean,
                                "Max_Error": maximum, "RMSE": rmse,
                                "Std_Error": std})
        algebraic_series.append(algebraic)
        mean, maximum, rmse, std = _error_summary(geometric)
        geometric_stats.append({"Method": label, "Mean_Error_m": mean,
                                "Max_Error_m": maximum, "RMSE_m": rmse,
                                "Std_Error_m": std})
        geometric_series.append(geometric)
        time = np.arange(len(algebraic)) * dt
        ax_alg.plot(time, algebraic, color=color, label=label, linewidth=3, alpha=0.8)
        ax_geo.plot(time, geometric, color=color, label=label, linewidth=3, alpha=0.8)
        np.save(os.path.join(output_folder, f"{label}_algebraic_errors.npy"), algebraic)
        np.save(os.path.join(output_folder, f"{label}_geometric_errors.npy"), geometric)
        plot_jsbsim_diagnostics(values, label, output_folder, dt)

    ax_alg.set(xlabel="Time (s)", ylabel="Algebraic Path Tracking Error (scaled residual; not m)")
    ax_alg.set_xlim(0, max_length * dt)
    ax_alg.legend()
    ax_alg.grid(True)
    _save_figure(fig_alg, os.path.join(output_folder, "algebraic_path_tracking_error"))

    ax_geo.set(xlabel="Time (s)", ylabel="Geometric Path Tracking Error (m)")
    ax_geo.set_xlim(0, max_length * dt)
    ax_geo.legend()
    ax_geo.grid(True)
    _save_figure(fig_geo, os.path.join(output_folder, "geometric_path_tracking_error"))

    pd.DataFrame(algebraic_stats).to_csv(
        os.path.join(output_folder, "algebraic_error_statistics.csv"), index=False)
    pd.DataFrame(geometric_stats).to_csv(
        os.path.join(output_folder, "geometric_error_statistics.csv"), index=False)

    max_n = max(map(len, algebraic_series))
    algebraic_matrix = np.full((max_n, len(algebraic_series)), np.nan)
    for index, values in enumerate(algebraic_series):
        algebraic_matrix[:len(values), index] = values
    pd.DataFrame(algebraic_matrix, columns=labels).to_csv(
        os.path.join(output_folder, "algebraic_errors_time_series.csv"),
        index_label="Time_Step")

    max_n = max(map(len, geometric_series))
    geometric_matrix = np.full((max_n, len(geometric_series)), np.nan)
    for index, values in enumerate(geometric_series):
        geometric_matrix[:len(values), index] = values
    pd.DataFrame(geometric_matrix, columns=labels).to_csv(
        os.path.join(output_folder, "geometric_errors_time_series.csv"),
        index_label="Time_Step")

    print("\n=== Algebraic Path Tracking Error Statistics ===")
    print(f"{'Method':<20} {'Mean':<10} {'Max':<10} {'RMSE':<10} {'Std':<10}")
    for row in algebraic_stats:
        print(f"{row['Method']:<20} {row['Mean_Error']:<10.4f} "
              f"{row['Max_Error']:<10.4f} {row['RMSE']:<10.4f} "
              f"{row['Std_Error']:<10.4f}")
    print("\n=== Euclidean Cross-Track Distance Statistics (m) ===")
    print(f"{'Method':<20} {'Mean':<10} {'Max':<10} {'RMSE':<10} {'Std':<10}")
    for row in geometric_stats:
        print(f"{row['Method']:<20} {row['Mean_Error_m']:<10.4f} "
              f"{row['Max_Error_m']:<10.4f} {row['RMSE_m']:<10.4f} "
              f"{row['Std_Error_m']:<10.4f}")
    save_localization_statistics(data_folders, labels, output_folder, max_length)
    print(f"Comparison plots saved to {output_folder}")


if __name__ == "__main__":
    path_type = "cassini"
    root = f"./jsbsim_data/{path_type}_path_following_data"
    methods = ["traditional", "cnn", "dp", "generative"]
    labels = ["Classical", "CNN-based", "Diffusion Policy", "Proposed GVC"]
    folders = [os.path.join(root, method) for method in methods]
    position_folder = os.path.join(root, "position_gvf")
    if os.path.isfile(os.path.join(position_folder, "positions.npy")):
        folders.append(position_folder)
        labels.append("VisLoc-GVF")
    output = "./comparison_plots_cassini_jsbsim"
    compare_jsbsim_data(folders, labels, output, max_length=2000, dt=DT)
