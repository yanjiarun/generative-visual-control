import argparse
import os


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "src", "path_following", "vector_field_quartic_obstacle.csv")
IMAGES = os.path.join(ROOT, "src", "path_following", "rectified_panorama_quartic_obstacle")


class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def train(kind, epochs, batch_size):
    if kind == "flow":
        from src.generative_model.core.generative_controller import train_flow_matching
        return train_flow_matching(Args(
            batch_size=batch_size, epochs=epochs, learning_rate=1e-4,
            vector_field_file=CSV, img_dir=IMAGES,
            save_dir=os.path.join(ROOT, "src", "generative_model", "models",
                                   "flow_matching_quartic_obstacle_models"),
            grad_clip=1.0, checkpoint_interval=50, dataset_type="normal",
            mask_type="none"))
    if kind == "cnn":
        from src.baseline.cnn_model import train_cnn_model
        return train_cnn_model(Args(
            batch_size=batch_size, epochs=epochs, learning_rate=1e-3,
            vector_field_file=CSV, img_dir=IMAGES,
            save_dir=os.path.join(ROOT, "src", "baseline", "models",
                                   "cnn_quartic_obstacle_models"),
            grad_clip=1.0, checkpoint_interval=50, dataset_type="normal",
            mask_type="none"))
    if kind == "dp":
        from src.generative_model.core.diffusion_policy import train_diffusion_policy
        return train_diffusion_policy(Args(
            vector_field_file=CSV, img_dir=IMAGES, data_dir=None,
            batch_size=batch_size, epochs=epochs, learning_rate=1e-4,
            seq_length=1, pred_horizon=2, num_workers=0, grad_clip=1.0,
            checkpoint_interval=50,
            save_dir=os.path.join(ROOT, "src", "generative_model", "models",
                                   "diffusion_policy_quartic_obstacle_models")))
    raise ValueError("kind must be flow, cnn, or dp")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", choices=("flow", "cnn", "dp"),
                   default=("flow", "cnn", "dp"))
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=32)
    a = p.parse_args()
    for method in a.methods:
        print(f"\n=== training {method} ===", flush=True)
        train(method, a.epochs, a.batch_size)
