import cv2
import numpy as np
import pandas as pd

from supplementary.gvf_diagnostics.settings import (
    BOUNDS, DATA, EPS, FD_STEP_M, GRID, LOCAL_POINTS, MIN_GRID_DISTANCE_M,
    ORIGIN, POINTS, SCALE, SEED, field, guard, sha256,
)


def make_manifest():
    grid = pd.read_csv(GRID)[["x", "y"]].to_numpy(float)
    rng = np.random.default_rng(SEED)
    local = set(rng.choice(POINTS, LOCAL_POINTS, replace=False))

    shape = np.array([20, int(np.ceil(POINTS / 20))])
    margin = FD_STEP_M / SCALE
    low, high = BOUNDS[:, 0] + margin, BOUNDS[:, 1] - margin
    offsets = {"center": np.zeros(2)}
    for tag, step in (("h", FD_STEP_M), ("half", FD_STEP_M / 2)):
        offsets.update({f"{tag}_{name}": step / SCALE * np.array(vector)
                        for name, vector in zip(("xp", "xm", "yp", "ym"),
                                                ((1, 0), (-1, 0), (0, 1), (0, -1)))})
    rows = []
    for sample, cell_id in enumerate(rng.permutation(np.prod(shape))[:POINTS]):
        cell = np.array([cell_id % shape[0], cell_id // shape[0]])
        views = offsets if sample in local else {"center": offsets["center"]}
        for _ in range(10000):
            point = low + (cell + rng.random(2)) * (high - low) / shape
            positions = np.array([point + offset for offset in views.values()])
            distances = np.linalg.norm(positions[:, None] - grid, axis=2).min(axis=1) * SCALE
            if distances.min() >= MIN_GRID_DISTANCE_M and all(np.linalg.norm(field(p)[:2]) > EPS for p in positions):
                break
        else:
            raise ValueError("Cannot sample the requested off-grid stencil")
        items = list(views.items())
        if sample in local:
            items.append(("repeat_center", np.zeros(2)))
        for view, offset in items:
            xy = point + offset
            world = ORIGIN + SCALE * xy
            rows.append({"capture_id": len(rows) + 1, "sample": sample + 1, "view": view,
                         "local": sample in local, "x_norm": xy[0], "y_norm": xy[1],
                         "x_world": world[0], "y_world": world[1],
                         "grid_distance_m": float(np.linalg.norm(grid - xy, axis=1).min() * SCALE),
                         "image": f"{sample + 1:03d}_{view}.png"})
    return pd.DataFrame(rows)


def main():
    images = DATA / "images"
    images.mkdir(parents=True, exist_ok=True)
    protocol = {"points": POINTS, "local_points": LOCAL_POINTS, "seed": SEED,
                "bounds": BOUNDS.tolist(), "fd_step_m": FD_STEP_M,
                "minimum_grid_distance_m": MIN_GRID_DISTANCE_M, "grid_sha256": sha256(GRID),
                "scale": SCALE, "origin": ORIGIN.tolist(), "altitude_m": 75.0,
                "camera": "experiment-1 world-aligned panorama; atomic six-camera capture"}
    guard(DATA / "protocol.json", protocol)
    table = make_manifest()
    path = DATA / "manifest.csv"
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(path), table, check_dtype=False, atol=1e-12, rtol=0)
    else:
        table.to_csv(path, index=False)
    from supplementary.visual_robustness.camera import Camera
    camera = None
    try:
        for row in table.itertuples():
            path = images / row.image
            if path.exists():
                if cv2.imread(str(path)) is None:
                    raise ValueError(f"Unreadable image: {path}")
                continue
            if camera is None:
                camera = Camera()
            image = camera.get_panoramic_image(row.capture_id, [row.x_world, row.y_world])
            temporary = path.with_suffix(".tmp.png")
            if not cv2.imwrite(str(temporary), image):
                raise OSError(f"Cannot save {path}")
            temporary.replace(path)
            print(f"Capture {row.capture_id}/{len(table)}: point {row.sample}, {row.view}", flush=True)
    finally:
        if camera is not None:
            camera.close()
    print(f"Observations: {DATA}")


if __name__ == "__main__":
    main()
