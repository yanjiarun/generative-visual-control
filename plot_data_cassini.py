import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from src.path_following.create_path_cassini_oval import compute_cassini_field

def load_flight_data(data_folder):
    """
    Load flight data from a folder
    Returns: positions, commands, yaw_deg
    """
    if os.path.exists(f"{data_folder}/poses.npy"):
        poses = np.load(f"{data_folder}/poses.npy")
        commands = np.load(f"{data_folder}/heading_commands.npy")
        positions = np.column_stack([poses[:, 0], poses[:, 1], np.full(len(poses), 100.0)])
        yaw_deg = np.degrees(poses[:, 2])
    else:
        positions = np.load(f"{data_folder}/positions.npy")
        commands = np.load(f"{data_folder}/commands.npy")
        atti_data = np.load(f"{data_folder}/atti_data.npy", allow_pickle=True)
        yaw_deg = np.degrees([a[2] for a in atti_data])
    
    return positions, commands, yaw_deg

def cassini_path_error(point, a=1.5, b=0.9, scale_factor=10):
    """Return the scaled Cassini residual at one point."""
    x, y = point
    alpha = (x**2 + y**2)**2 - 2*a**2*(x**2 - y**2) - b**4
    return abs(alpha * scale_factor)

def calculate_algebraic_path_errors(positions, base_x=576, base_y=275, scale_factor=100, a=1.5, b=0.9):
    """Return algebraic Cassini errors for world-coordinate positions."""
    errors = []
    for pos in positions:
        x_norm = (pos[0] - base_x) / scale_factor
        y_norm = (pos[1] - base_y) / scale_factor
        error = cassini_path_error((x_norm, y_norm), a, b)
        errors.append(error)
    
    return np.array(errors)


def build_cassini_zero_contour(a=1.5, b=0.9, limits=(-2.8, 2.8, -1.4, 1.4), resolution=(1201, 601)):
    """Extract high-resolution alpha=0 contour components in normalized coordinates."""
    xmin, xmax, ymin, ymax = limits
    nx, ny = resolution
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)
    _, _, alpha = compute_cassini_field(X, Y, a, b)
    fig, ax = plt.subplots()
    try:
        segments = [s for s in ax.contour(X, Y, alpha, levels=[0]).allsegs[0] if len(s) > 1]
    finally:
        plt.close(fig)
    if not segments:
        raise RuntimeError("Cassini alpha=0 contour is empty")
    return segments


def point_to_contour_distance(points, contour_segments):
    """Return point-to-polyline distances in the input coordinate units."""
    points = np.asarray(points, dtype=float)
    starts, ends = [], []
    for segment in contour_segments:
        segment = np.asarray(segment, dtype=float)
        if len(segment) < 2:
            continue
        if not np.allclose(segment[0], segment[-1]):
            segment = np.vstack((segment, segment[0]))
        starts.append(segment[:-1])
        ends.append(segment[1:])
    if not starts:
        raise ValueError("No valid contour segments")
    starts, vectors = np.concatenate(starts), np.concatenate(ends) - np.concatenate(starts)
    lengths = np.sum(vectors * vectors, axis=1)
    keep = lengths > np.finfo(float).eps
    starts, vectors, lengths = starts[keep], vectors[keep], lengths[keep]
    relative = points[:, None, :] - starts[None, :, :]
    t = np.sum(relative * vectors[None, :, :], axis=2) / lengths[None, :]
    t = np.clip(t, 0, 1)
    relative -= t[:, :, None] * vectors[None, :, :]
    return np.sqrt(np.min(np.sum(relative * relative, axis=2), axis=1))


def calculate_geometric_path_errors(
    positions, contour_segments=None, base_x=576, base_y=275, scale_factor=100,
    a=1.5, b=0.9, contour_limits=(-2.8, 2.8, -1.4, 1.4), contour_resolution=(1201, 601)
):
    """Calculate Euclidean cross-track errors in metres."""
    if contour_segments is None:
        contour_segments = build_cassini_zero_contour(a, b, contour_limits, contour_resolution)
    normalized = (np.asarray(positions)[:, :2] - [base_x, base_y]) / scale_factor
    return scale_factor * point_to_contour_distance(normalized, contour_segments)


def _error_summary(errors, start=500):
    values = np.asarray(errors, dtype=float)
    if len(values) > start:
        values = values[start:]
    return np.mean(values), np.max(values), np.sqrt(np.mean(values**2)), np.std(values)


def save_localization_statistics(data_folders, labels, output_folder, max_length=2000):
    rows = []
    for folder, label in zip(data_folders, labels):
        path = os.path.join(folder, "position_errors_m.npy")
        if not os.path.isfile(path):
            continue
        errors = np.load(path)
        if max_length is not None:
            errors = errors[:max_length]
        if not len(errors):
            continue
        row = {"Method": label, "Frames": len(errors),
               "Position_MAE_m": float(np.mean(errors)),
               "Position_RMSE_m": float(np.sqrt(np.mean(errors ** 2))),
               "Position_Max_Error_m": float(np.max(errors))}
        for name, prefix in (("cnn_inference_times_ms", "CNN_Inference"),
                             ("position_gvf_times_ms", "Position_GVF")):
            path = os.path.join(folder, f"{name}.npy")
            if not os.path.isfile(path):
                continue
            times = np.load(path)[:len(errors)]
            start = 20 if len(times) > 20 else 0
            times = times[start:]
            if len(times):
                row[f"{prefix}_Mean_ms"] = float(np.mean(times))
                row[f"{prefix}_P95_ms"] = float(np.percentile(times, 95))
                row["Timing_Warmup_Frames"] = start
        rows.append(row)
    if rows:
        os.makedirs(output_folder, exist_ok=True)
        pd.DataFrame(rows).to_csv(
            os.path.join(output_folder, "position_localization_inference_statistics.csv"), index=False)


def compare_multiple_data(data_folders, labels=None, path_type="none", max_length=None, init_pose=None):
    """
    Create comparison plots for multiple flight datasets
    :param data_folders: List of data folder paths
    :param labels: List of labels for each dataset
    :param output_folder: Output directory for plots
    """
    if labels is None:
        labels = [os.path.basename(folder) for folder in data_folders]

    output_folder = f"./comparison_plots_{path_type}_{init_pose[0]}_{init_pose[1]}_{int(init_pose[2]*180/np.pi)}" if path_type != "none" else "./comparison_plots"
    
    os.makedirs(output_folder, exist_ok=True)
    
    all_data = []
    for folder in data_folders:
        positions, commands, yaw_deg = load_flight_data(folder)
        if max_length:
            n = min(len(positions), max_length)
            positions, commands, yaw_deg = positions[:n], commands[:n], yaw_deg[:n]
        all_data.append({
            'positions': positions,
            'commands': commands,
            'yaw_deg': yaw_deg
        })
    
    time = np.arange(len(all_data[0]['positions']))
    
    colors = ['gray', 'darkblue', 'darkorange', 'darkred', 'purple']
    
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 24

    plt.figure(figsize=(12, 10))
    
    for i, data in enumerate(all_data):
        plt.plot(data['positions'][:, 0], data['positions'][:, 1], 
                color=colors[i], label=labels[i], linestyle='-', linewidth=3, alpha=0.8)
    if path_type == "cassini":
        n_points = 200
        x_norm = np.linspace(-2.8, 2.8, n_points)
        y_norm = np.linspace(-1.4, 1.4, n_points)
        X_norm, Y_norm = np.meshgrid(x_norm, y_norm)
        
        _, _, alpha = compute_cassini_field(X_norm, Y_norm)
        
        base_x, base_y = 576, 275  # Base coordinates in Unity
        scale_factor = 100  # From collect_imgs_path_following.py
        
        X_world = base_x + X_norm * scale_factor
        Y_world = base_y + Y_norm * scale_factor
        
        cs = plt.contour(X_world, Y_world, alpha, levels=[0], colors='darkgreen', linewidths=5, linestyles='--')
        
        plt.plot([], [], color='darkgreen', linestyle='--', linewidth=5, label='Desired Curve')

    plt.axis('scaled')
    plt.xlim(330, 820)
    plt.ylim(90, 490)
    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_folder}/trajectory_comparison_{path_type}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_folder}/trajectory_comparison_{path_type}.pdf", bbox_inches='tight')
    plt.show()
    
    if path_type == "cassini":
        error_stats = []
        geometric_stats = []
        geometric_contour = build_cassini_zero_contour()
        
        plt.figure(figsize=(12, 8))
        
        for i, data in enumerate(all_data):
            algebraic_errors = calculate_algebraic_path_errors(data['positions'])
            
            if len(algebraic_errors) > 500:  # Use the converged tail for statistics.
                converged_errors = algebraic_errors[500:]
                mean_error = np.mean(converged_errors)
                max_error = np.max(converged_errors)
                rmse = np.sqrt(np.mean(converged_errors**2))
                std_error = np.std(converged_errors)
            else:
                mean_error = np.mean(algebraic_errors)
                max_error = np.max(algebraic_errors)
                rmse = np.sqrt(np.mean(algebraic_errors**2))
                std_error = np.std(algebraic_errors)
            
            error_stats.append({
                'method': labels[i],
                'mean': mean_error,
                'max': max_error,
                'rmse': rmse,
                'std': std_error,
                'errors': algebraic_errors
            })

            # True Euclidean cross-track error in metres.
            geometric_errors = calculate_geometric_path_errors(
                data['positions'], contour_segments=geometric_contour
            )
            mean_error, max_error, rmse, std_error = _error_summary(geometric_errors)
            geometric_stats.append({
                'method': labels[i],
                'mean': mean_error,
                'max': max_error,
                'rmse': rmse,
                'std': std_error,
                'errors': geometric_errors
            })
            
            time_seconds = np.arange(len(algebraic_errors)) * 0.05  # Assuming dt=0.05s
            plt.plot(time_seconds, algebraic_errors, color=colors[i], 
                    label=f'{labels[i]}', linewidth=3, alpha=0.8)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Algebraic Path Tracking Error (scaled residual; not m)')
        plt.legend()
        plt.xlim(0, 100)
        plt.grid(True)
        plt.savefig(f"{output_folder}/algebraic_path_tracking_error.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_folder}/algebraic_path_tracking_error.pdf", bbox_inches='tight')
        plt.show()
        
        print("\n=== Algebraic Path Tracking Error Statistics ===")
        print(f"{'Method':<15} {'Mean':<10} {'Max':<10} {'RMSE':<10} {'Std':<10}")
        print("-" * 60)
        for stats in error_stats:
            print(f"{stats['method']:<15} {stats['mean']:<10.4f} {stats['max']:<10.4f} {stats['rmse']:<10.4f} {stats['std']:<10.4f}")
        
        error_data = []
        for stats in error_stats:
            error_data.append({
                'Method': stats['method'],
                'Mean_Error': stats['mean'],
                'Max_Error': stats['max'],
                'RMSE': stats['rmse'],
                'Std_Error': stats['std']
            })
        
        df_stats = pd.DataFrame(error_data)
        df_stats.to_csv(f"{output_folder}/algebraic_error_statistics.csv", index=False)
        
        for i, stats in enumerate(error_stats):
            np.save(f"{output_folder}/{stats['method']}_algebraic_errors.npy", stats['errors'])
        
        max_length = max(len(stats['errors']) for stats in error_stats)
        error_matrix = np.full((max_length, len(error_stats)), np.nan)
        
        for i, stats in enumerate(error_stats):
            error_matrix[:len(stats['errors']), i] = stats['errors']
        
        df_errors = pd.DataFrame(error_matrix, columns=[stats['method'] for stats in error_stats])
        df_errors.to_csv(f"{output_folder}/algebraic_errors_time_series.csv", index_label='Time_Step')
        
        print(f"\nError data saved to:")
        print(f"- {output_folder}/algebraic_error_statistics.csv")
        print(f"- {output_folder}/algebraic_errors_time_series.csv")
        for stats in error_stats:
            print(f"- {output_folder}/{stats['method']}_algebraic_errors.npy")

        plt.figure(figsize=(12, 8))
        for i, stats in enumerate(geometric_stats):
            time_seconds = np.arange(len(stats['errors'])) * 0.05
            plt.plot(time_seconds, stats['errors'], color=colors[i],
                     label=f'{labels[i]}', linewidth=3, alpha=0.8)
        plt.xlabel('Time (s)')
        plt.ylabel('Geometric Path Tracking Error (m)')
        plt.legend()
        plt.xlim(0, 100)
        plt.grid(True)
        plt.savefig(f"{output_folder}/geometric_path_tracking_error.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_folder}/geometric_path_tracking_error.pdf", bbox_inches='tight')
        plt.show()

        print("\n=== Euclidean Cross-Track Distance Statistics (m) ===")
        print(f"{'Method':<15} {'Mean':<10} {'Max':<10} {'RMSE':<10} {'Std':<10}")
        print("-" * 60)
        for stats in geometric_stats:
            print(f"{stats['method']:<15} {stats['mean']:<10.4f} {stats['max']:<10.4f} "
                  f"{stats['rmse']:<10.4f} {stats['std']:<10.4f}")

        geometric_data = [{
            'Method': stats['method'],
            'Mean_Error_m': stats['mean'],
            'Max_Error_m': stats['max'],
            'RMSE_m': stats['rmse'],
            'Std_Error_m': stats['std']
        } for stats in geometric_stats]
        pd.DataFrame(geometric_data).to_csv(
            f"{output_folder}/geometric_error_statistics.csv", index=False
        )
        for stats in geometric_stats:
            np.save(f"{output_folder}/{stats['method']}_geometric_errors.npy", stats['errors'])
        max_length = max(len(stats['errors']) for stats in geometric_stats)
        geometric_matrix = np.full((max_length, len(geometric_stats)), np.nan)
        for i, stats in enumerate(geometric_stats):
            geometric_matrix[:len(stats['errors']), i] = stats['errors']
        pd.DataFrame(
            geometric_matrix, columns=[stats['method'] for stats in geometric_stats]
        ).to_csv(
            f"{output_folder}/geometric_errors_time_series.csv", index_label='Time_Step'
        )
        print(f"- {output_folder}/geometric_error_statistics.csv")
        print(f"- {output_folder}/geometric_errors_time_series.csv")
        for stats in geometric_stats:
            print(f"- {output_folder}/{stats['method']}_geometric_errors.npy")
    
    save_localization_statistics(data_folders, labels, output_folder, max_length)
    print(f"\nComparison plots saved to {output_folder}")

if __name__ == "__main__":
    PATH_TYPE = "cassini"
    init_pose = np.array([576, 275, np.pi/3])
    data_folders = [
        f"./data/{PATH_TYPE}_path_following_data/{method}"
        for method in ("traditional", "cnn", "dp", "generative")
    ]
    labels = ["Classical", "CNN-based", "Diffusion Policy", "Proposed GVC"]
    position_folder = f"./data/{PATH_TYPE}_path_following_data/position_gvf"
    if os.path.isfile(os.path.join(position_folder, "poses.npy")):
        data_folders.append(position_folder)
        labels.append("VisLoc-GVF")
    compare_multiple_data(data_folders, labels=labels, path_type=PATH_TYPE,
                          max_length=2000, init_pose=init_pose)
