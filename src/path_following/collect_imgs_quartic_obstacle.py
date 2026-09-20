import argparse
import os

import pandas as pd

from AFPV_Kit.common.state import PoseState
from AFPV_Kit.common.transfer import convert_ned_quaternion_to_enu
from src.path_following.create_path_quartic_obstacle import SCALE_FACTOR
from src.utils.collect_imgs import ImageCollector
from src.utils.rectify_imgs import rectify_folder


def collect(csv_file, output_dir, rectified_dir=None, altitude=75.0):
    table = pd.read_csv(csv_file).drop_duplicates("frame_id")
    collector = ImageCollector(output_dir, debug=True)
    q = convert_ned_quaternion_to_enu([0, 0, 0, 1])
    try:
        for i, row in table.iterrows():
            frame = int(row.frame_id)
            pos = [576.0 + float(row.x) * SCALE_FACTOR,
                   275.0 + float(row.y) * SCALE_FACTOR, altitude]
            print(f"[{i + 1}/{len(table)}] frame {frame}: {pos[:2]}")
            collector.get_panorama(frame, PoseState(pos, q), save_color=True)
    finally:
        collector.unity_bridge.close()
    if rectified_dir:
        rectify_folder(output_dir, rectified_dir, color_mode=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="./src/path_following/vector_field_quartic_obstacle.csv")
    p.add_argument("--output", default="./src/path_following/panorama_quartic_obstacle")
    p.add_argument("--rectified-output", default="./src/path_following/rectified_panorama_quartic_obstacle")
    p.add_argument("--altitude", type=float, default=75.0)
    a = p.parse_args()
    os.makedirs(a.output, exist_ok=True)
    collect(a.csv, a.output, a.rectified_output, a.altitude)
