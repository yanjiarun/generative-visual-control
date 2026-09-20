# Experiment 4: Real-Time Benchmark

This directory contains standalone benchmark code. Before running it, provide:

```text
inputs/rectified_color_panorama_0001.jpg
weights/cnn_cassini.pth
weights/diffusion_policy_cassini.pth
weights/flow_matching_cassini.pth
```

TensorRT engines are generated locally in `engines/` on the target Orin device.

## Scripts

- `models.py`: standalone CNN, Diffusion Policy, and GVC model definitions.
- `benchmark_realtime_pc.py`: PyTorch benchmark for a PC or Orin GPU.
- `benchmark_realtime_orin.py`: Orin NX PyTorch end-to-end benchmark.
- `export_tensorrt_orin.py`: export CNN, DP, and GVC components to TensorRT engines.
- `benchmark_realtime_orin_tensorrt.py`: Orin NX TensorRT end-to-end benchmark.
- `run_memory_tests_orin.sh`: collect Orin RAM usage with `tegrastats` and Docker.
- `run_realtime_orin.sh` and `run_realtime_orin_tensorrt.sh`: short Orin launchers.

## PC PyTorch

Run from this directory after activating the environment:

```powershell
conda activate rainbow_sim
python .\benchmark_realtime_pc.py `
    --device cuda `
    --ode-steps 10 20 50 100 `
    --warmup 20 `
    --iterations 100 `
    --output results_pc_pytorch.csv
```

Use `--device cpu` for a CPU run. Add `--include-preprocess` to include image
resize and crop time. `--methods` selects any of `traditional cnn dp gvc`.
`--ode-steps` controls the DP/GVC sampling-step sweep.

## Orin PyTorch

```bash
python3 benchmark_realtime_orin.py \
    --device cuda \
    --ode-steps 10 20 50 100 \
    --warmup 20 \
    --iterations 100 \
    --output results_orin_pytorch.csv
```

## Orin TensorRT

Export engines on the target Orin device:

```bash
python3 export_tensorrt_orin.py --precision fp16
```

Then benchmark the complete sampling loops:

```bash
python3 benchmark_realtime_orin_tensorrt.py \
    --methods traditional cnn dp gvc \
    --ode-steps 10 20 50 100 \
    --warmup 20 \
    --iterations 100 \
    --output results_orin_tensorrt.csv
```

Use `--precision fp32` when exporting a FP32 engine.

## Docker on Orin

Build the image on Orin NX:

```bash
docker build --network host -t orin-realtime:jp62 .
```

Run the PyTorch or TensorRT launcher from this directory:

```bash
docker run --rm --runtime nvidia --network host \
    -v "$PWD:/workspace" -w /workspace orin-realtime:jp62 \
    bash run_realtime_orin.sh

docker run --rm --runtime nvidia --network host \
    -v "$PWD:/workspace" -w /workspace orin-realtime:jp62 \
    bash run_realtime_orin_tensorrt.sh
```

For RAM profiling, run `bash run_memory_tests_orin.sh` on the Orin host. It
requires `tegrastats`, Docker, and the TensorRT engines for the TensorRT cases.
