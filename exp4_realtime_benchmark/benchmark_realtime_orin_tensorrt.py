import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tensorrt as trt
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from benchmark_realtime_pc import INPUT, cassini_traditional_step, load_tensor, measure


ROOT = Path(__file__).resolve().parent
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def trt_to_torch(dtype):
    table = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT8: torch.int8,
        trt.DataType.INT32: torch.int32,
        trt.DataType.BOOL: torch.bool,
    }
    if hasattr(trt.DataType, "INT64"):
        table[trt.DataType.INT64] = torch.int64
    if dtype not in table:
        raise TypeError(f"Unsupported TensorRT dtype: {dtype}")
    return table[dtype]


class TRTModule:
    def __init__(self, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        self.runtime = trt.Runtime(TRT_LOGGER)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {path}")

        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs = [], []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(name)
            else:
                self.outputs.append(name)

    def __call__(self, *args):
        if len(args) != len(self.inputs):
            raise ValueError(f"Expected {len(self.inputs)} inputs, got {len(args)}")

        for name, x in zip(self.inputs, args):
            expected_shape = tuple(self.engine.get_tensor_shape(name))
            expected_dtype = trt_to_torch(self.engine.get_tensor_dtype(name))

            if tuple(x.shape) != expected_shape:
                raise ValueError(
                    f"{name}: engine expects shape {expected_shape}, got {tuple(x.shape)}"
                )
            if x.dtype != expected_dtype:
                raise TypeError(
                    f"{name}: engine expects {expected_dtype}, got {x.dtype}. "
                    "Re-export the engine with the correct input dtype."
                )
            if not x.is_cuda or not x.is_contiguous():
                raise ValueError(f"{name} must be a contiguous CUDA tensor")

            self.context.set_tensor_address(name, x.data_ptr())

        outputs = []
        for name in self.outputs:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = trt_to_torch(self.engine.get_tensor_dtype(name))
            y = torch.empty(shape, device="cuda", dtype=dtype)
            self.context.set_tensor_address(name, y.data_ptr())
            outputs.append(y)

        ok = self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")

        return outputs[0] if len(outputs) == 1 else outputs


def measure_cuda(fn, warmup, iterations, stream):
    with torch.inference_mode():
        torch.cuda.synchronize()

        with torch.cuda.stream(stream):
            for _ in range(warmup):
                fn()
        stream.synchronize()

        elapsed = []
        for _ in range(iterations):
            stream.synchronize()
            start = time.perf_counter()

            with torch.cuda.stream(stream):
                fn()

            stream.synchronize()
            elapsed.append((time.perf_counter() - start) * 1000.0)

    v = np.asarray(elapsed, dtype=np.float64)
    return {
        "iterations": int(iterations),
        "mean_ms": float(v.mean()),
        "median_ms": float(np.median(v)),
        "p95_ms": float(np.percentile(v, 95)),
        "min_ms": float(v.min()),
        "max_ms": float(v.max()),
        "fps": float(1000.0 / v.mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+",
                   default=["traditional", "cnn", "dp", "gvc"],
                   choices=["traditional", "cnn", "dp", "gvc"])
    p.add_argument("--engines", type=Path, default=ROOT / "engines")
    p.add_argument("--input", type=Path, default=INPUT)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--traditional-iterations", type=int, default=10000)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--ode-steps", nargs="+", type=int,
                   default=[10, 20, 50, 100])
    p.add_argument("--output", type=Path,
                   default=ROOT / "results_orin_tensorrt_e2e.csv")
    args = p.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    device = torch.device("cuda")
    stream = torch.cuda.Stream()

    # Same as the PC default benchmark:
    # disk I/O and resize/crop preprocessing are NOT timed.
    image, _, _ = load_tensor(args.input, device)

    # TensorRT requires contiguous input buffers.
    # Do this once before timing, so the copy is NOT counted in inference latency.
    image = image.contiguous()
    image_seq = image[:, None].contiguous()

    torch.cuda.synchronize()

    rows = []

    def add(method, timing, steps=None, timing_device="cuda"):
        row = {
            "backend": "tensorrt",
            "device": timing_device,
            "method": method,
            "ode_steps": steps if steps is not None else "",
        }
        row.update(timing)
        rows.append(row)

    if "traditional" in args.methods:
        add(
            "traditional",
            measure(
                lambda: cassini_traditional_step(),
                torch.device("cpu"),
                args.warmup,
                args.traditional_iterations,
            ),
            timing_device="cpu",
        )

    if "cnn" in args.methods:
        cnn = TRTModule(args.engines / "cnn.engine")
        add(
            "cnn",
            measure_cuda(
                lambda: cnn(image),
                args.warmup,
                args.iterations,
                stream,
            ),
        )

    if "gvc" in args.methods:
        gvc_encoder = TRTModule(args.engines / "gvc_encoder.engine")
        gvc_step = TRTModule(args.engines / "gvc_step.engine")

        for steps in args.ode_steps:

            def gvc_call(steps=steps):
                features = gvc_encoder(image_seq)
                action = torch.rand(
                    1, 4,
                    device=device,
                    dtype=torch.float32
                ) * 2.0 - 1.0

                dt = 1.0 / steps
                for i in range(steps):
                    t = torch.full(
                        (1, 1),
                        i / steps,
                        device=device,
                        dtype=torch.float32
                    )
                    velocity = gvc_step(features, action, t)
                    action = action + velocity * dt
                return action

            add(
                "gvc",
                measure_cuda(
                    gvc_call,
                    args.warmup,
                    args.iterations,
                    stream,
                ),
                steps,
            )

    if "dp" in args.methods:
        dp_encoder = TRTModule(args.engines / "dp_encoder.engine")
        dp_step = TRTModule(args.engines / "dp_step.engine")

        scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_start=1e-4,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            variance_type="fixed_small",
            clip_sample=True,
            prediction_type="epsilon",
        )

        for steps in args.ode_steps:
            def dp_call(steps=steps):
                # Full pipeline: image -> encoder -> N DDPM steps -> final action
                features = dp_encoder(image_seq)
                action = torch.randn(
                    1, 2, 2, device=device, dtype=torch.float32
                )

                scheduler.set_timesteps(steps)

                for t in scheduler.timesteps:
                    ts = torch.full(
                        (1,),
                        int(t),
                        device=device,
                        dtype=torch.long,
                    )
                    noise = dp_step(action, ts, features)
                    action = scheduler.step(
                        noise, t, action
                    ).prev_sample

                return F.normalize(action, dim=-1)

            add(
                "dp",
                measure_cuda(
                    dp_call,
                    args.warmup,
                    args.iterations,
                    stream,
                ),
                steps,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    for row in rows:
        suffix = f", steps={row['ode_steps']}" if row["ode_steps"] else ""
        print(
            f"{row['method']:12s}{suffix:12s} "
            f"{row['mean_ms']:.3f} ms "
            f"(p95 {row['p95_ms']:.3f}, {row['fps']:.1f} FPS)"
        )

    print(f"Saved {args.output} and {json_path}")


if __name__ == "__main__":
    main()
