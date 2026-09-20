import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from models import CNN, DiffusionPolicy, FlowPolicy


ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights"
INPUT = ROOT / "inputs" / "rectified_color_panorama_0001.jpg"


def load_tensor(path, device, include_preprocess=False):
    image = Image.open(path).convert("RGB")
    prep = transforms.Compose([transforms.Resize((256, 256)),
                               transforms.CenterCrop((224, 224))])
    image = prep(image)
    x = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)
    x = x.float().div(255).unsqueeze(0).to(device)
    return x, prep, Image.open(path).convert("RGB") if include_preprocess else None


def load_state(model, path, device):
    state = torch.load(path, map_location=device)
    model.to(device).load_state_dict(state.get("model_state_dict", state), strict=True)
    return model.eval()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(fn, device, warmup, iterations):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        sync(device)
        elapsed = []
        for _ in range(iterations):
            sync(device)
            start = time.perf_counter()
            fn()
            sync(device)
            elapsed.append((time.perf_counter() - start) * 1000.0)
    values = np.asarray(elapsed, dtype=np.float64)
    return {
        "iterations": int(iterations), "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()), "max_ms": float(values.max()),
        "fps": float(1000.0 / values.mean()),
    }


def cassini_traditional_step(position=(600.0, 300.0), heading=0.4):
    """Scalar Cassini GVF + heading update, representative CPU baseline."""
    x, y = (position[0] - 576.0) / 100.0, (position[1] - 275.0) / 100.0
    a, b = 1.5, 0.9
    alpha = (x * x + y * y) ** 2 - 2 * a * a * (x * x - y * y) - b ** 4
    dx = 4 * x * (x * x + y * y) - 4 * a * a * x
    dy = 4 * y * (x * x + y * y) + 4 * a * a * y
    p = 0.5 * alpha * alpha
    g = -(2 / math.pi) * math.atan(2 * math.sqrt(p))
    h = math.sqrt(max(0.0, 1.0 - g * g))
    n1 = max(math.hypot(alpha * dx, alpha * dy), 1e-8)
    n2 = max(math.hypot(dy, dx), 1e-8)
    ux, uy = g * alpha * dx / n1 - h * dy / n2, g * alpha * dy / n1 + h * dx / n2
    error = math.atan2(math.sin(math.atan2(uy, ux) - heading),
                       math.cos(math.atan2(uy, ux) - heading))
    return heading + 28.0 * 0.18 * math.sin(error)


def run_torch(args):
    requested = args.device
    device = (torch.device("cuda") if requested == "auto" and torch.cuda.is_available()
              else torch.device("cpu") if requested == "auto" else torch.device(requested))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is unavailable")
    image, prep, raw_image = load_tensor(args.input, device, args.include_preprocess)
    def image_input():
        if args.include_preprocess:
            x = torch.from_numpy(np.array(prep(raw_image), copy=True)).permute(2, 0, 1)
            return x.float().div(255).unsqueeze(0).to(device)
        return image

    results = []

    def add(method, timing, steps=None, timing_device=None):
        row = {"backend": "torch",
               "device": str(timing_device or device), "method": method,
               "ode_steps": steps if steps is not None else ""}
        row.update(timing)
        results.append(row)

    if "traditional" in args.methods:
        add("traditional", measure(lambda: cassini_traditional_step(),
                                    torch.device("cpu"), args.warmup,
                                    args.traditional_iterations),
            timing_device="cpu")

    if "cnn" in args.methods:
        model = load_state(CNN(), WEIGHTS / "cnn_cassini.pth", device)
        if args.include_preprocess:
            def cnn_call():
                model(image_input())
        else:
            cnn_call = lambda: model(image)
        add("cnn", measure(cnn_call, device, args.warmup, args.iterations))

    if "gvc" in args.methods:
        model = load_state(FlowPolicy(), WEIGHTS / "flow_matching_cassini.pth", device)
        for steps in args.ode_steps:
            def gvc_call(steps=steps):
                sequence = image_input()[:, None]
                features = model.encode(sequence)
                action = torch.rand(1, 4, device=device) * 2 - 1
                dt = 1.0 / steps
                for i in range(steps):
                    t = torch.full((1, 1), i / steps, device=device)
                    action = action + model.forward_cached(features, action, t) * dt
                return action
            add("gvc", measure(gvc_call, device, args.warmup, args.iterations), steps)

    if "dp" in args.methods:
        model = load_state(DiffusionPolicy(), WEIGHTS / "diffusion_policy_cassini.pth", device)
        scheduler = DDPMScheduler(num_train_timesteps=100, beta_start=1e-4,
                                  beta_end=0.02, beta_schedule="squaredcos_cap_v2",
                                  variance_type="fixed_small", clip_sample=True,
                                  prediction_type="epsilon")
        for steps in args.ode_steps:
            def dp_call(steps=steps):
                sequence = image_input()[:, None]
                features = model.encode(sequence)
                action = torch.randn(1, 2, 2, device=device)
                scheduler.set_timesteps(steps)
                for t in scheduler.timesteps:
                    ts = torch.full((1,), t, dtype=torch.long, device=device)
                    noise = model.forward_cached(action, ts, features)
                    action = scheduler.step(noise, t, action).prev_sample
                return F.normalize(action, dim=-1)
            add("dp", measure(dp_call, device, args.warmup, args.iterations), steps)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=["traditional", "cnn", "dp", "gvc"],
                   choices=["traditional", "cnn", "dp", "gvc"])
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--input", type=Path, default=INPUT)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--traditional-iterations", type=int, default=10000)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--ode-steps", nargs="+", type=int, default=[10, 20, 50, 100])
    p.add_argument("--include-preprocess", action="store_true")
    p.add_argument("--output", type=Path, default=ROOT / "results_pc.csv")
    args = p.parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    results = run_torch(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)
    args.output.with_suffix(".json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for row in results:
        suffix = f", ODE={row['ode_steps']}" if row["ode_steps"] else ""
        print(f"{row['method']:12s}{suffix:10s} {row['mean_ms']:.3f} ms "
              f"(p95 {row['p95_ms']:.3f}, {row['fps']:.1f} FPS)")
    print(f"Saved {args.output} and {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
