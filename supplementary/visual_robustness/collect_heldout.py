import json

import cv2
import numpy as np
import pandas as pd

from supplementary.visual_robustness.common import BASE_SEED, OFFLINE_COUNT, OFFLINE_DIR, ROOT, write_json
from supplementary.visual_robustness.camera import Camera

MIN_GRID_DISTANCE = 0.012  # Normalized coordinates (1.2 m).


def sample_points():
    rng = np.random.default_rng(BASE_SEED)
    grid = pd.read_csv(ROOT / "src/path_following/vector_field_with_gradients_cassini.csv")[["x", "y"]].to_numpy()

    points = []
    nx = 20
    ny = int(np.ceil(OFFLINE_COUNT / nx))
    for index in rng.permutation(nx * ny)[:OFFLINE_COUNT]:
        cell = np.array([index % nx, index // nx])
        while True:
            point = [-2.8, -1.4] + (cell + rng.random(2)) * [5.6 / nx, 2.8 / ny]
            if np.linalg.norm(grid - point, axis=1).min() > MIN_GRID_DISTANCE:
                points.append(point)
                break
    return np.asarray(points)


def main():
    image_dir = OFFLINE_DIR / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    points = sample_points()
    cfg = {"count": OFFLINE_COUNT, "seed": BASE_SEED, "minimum_grid_distance": MIN_GRID_DISTANCE,
           "sampling": "stratified continuous positions; 20 columns", "altitude_m": 75,
           "panorama": "experiment-1 world-aligned 270x1800; integer nearest remap"}
    cfg_path = OFFLINE_DIR / "sampling_config.json"
    if cfg_path.exists() and json.loads(cfg_path.read_text(encoding="utf-8")) != cfg:
        raise ValueError("Held-out protocol changed; choose a new OFFLINE_DIR")
    write_json(cfg_path, cfg)
    manifest = OFFLINE_DIR / "heldout_manifest.csv"
    if not manifest.exists():
        pd.DataFrame({"sample": np.arange(1, len(points) + 1), "x_norm": points[:, 0], "y_norm": points[:, 1],
                      "x_world": 576 + 100 * points[:, 0], "y_world": 275 + 100 * points[:, 1],
                      "image": [f"panorama_{i:04d}.png" for i in range(1, len(points) + 1)]}).to_csv(manifest, index=False)
    table = pd.read_csv(manifest)
    if len(table) != OFFLINE_COUNT or not np.allclose(table[["x_norm", "y_norm"]], points):
        raise ValueError("Saved held-out manifest does not match sampling protocol")
    camera = Camera()
    try:
        for row in table.itertuples():
            path = image_dir / row.image
            if path.exists():
                if cv2.imread(str(path)) is None:
                    raise ValueError(f"Invalid saved image: {path}")
                continue
            image = camera.get_panoramic_image(row.sample, [row.x_world, row.y_world])
            if not cv2.imwrite(str(path), image):
                raise OSError(f"Cannot save {path}")
            print(f"Held-out image {row.sample}/{OFFLINE_COUNT}")
    finally:
        camera.close()
    print(f"Held-out images: {image_dir}")


if __name__ == "__main__":
    main()
