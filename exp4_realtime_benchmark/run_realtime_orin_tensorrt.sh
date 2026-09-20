#!/bin/bash
python3 benchmark_realtime_orin_tensorrt.py \
  --methods traditional cnn dp gvc \
  --ode-steps 10 20 50 100 \
  --warmup 20 \
  --iterations 100 \
  --output results_orin_tensorrt.csv
