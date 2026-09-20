#!/usr/bin/env bash
set -euo pipefail

# One-click Jetson Orin memory benchmark
# Tests:
#   PyTorch / TensorRT
#   CNN
#   GVC: 10,20,50,100 steps
#   DP : 10,20,50,100 steps
#
# Output:
#   memory_benchmark/memory_results.csv
#   memory_benchmark/logs/*.log

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-orin-realtime:jp62}"
PYTORCH_SCRIPT="${PYTORCH_SCRIPT:-benchmark_realtime_orin.py}"
TRT_SCRIPT="${TRT_SCRIPT:-benchmark_realtime_orin_tensorrt.py}"

# Memory peak does not need many timing iterations.
WARMUP="${WARMUP:-5}"
ITERATIONS="${ITERATIONS:-20}"
SAMPLE_MS="${SAMPLE_MS:-100}"
COOLDOWN_SEC="${COOLDOWN_SEC:-1}"

OUTDIR="$ROOT/memory_benchmark"
LOGDIR="$OUTDIR/logs"
RESULT="$OUTDIR/memory_results.csv"

mkdir -p "$LOGDIR"

if ! command -v tegrastats >/dev/null 2>&1; then
    echo "ERROR: tegrastats not found. Run this script on the Jetson host."
    exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker not found."
    exit 1
fi

if [[ ! -f "$ROOT/$PYTORCH_SCRIPT" ]]; then
    echo "ERROR: missing $ROOT/$PYTORCH_SCRIPT"
    exit 1
fi

if [[ ! -f "$ROOT/$TRT_SCRIPT" ]]; then
    echo "ERROR: missing $ROOT/$TRT_SCRIPT"
    exit 1
fi

echo "backend,method,steps,baseline_ram_mb,peak_ram_mb,delta_ram_mb" > "$RESULT"

TG_PID=""

cleanup() {
    if [[ -n "${TG_PID:-}" ]]; then
        kill "$TG_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

start_tegrastats() {
    local logfile="$1"
    rm -f "$logfile"
    tegrastats --interval "$SAMPLE_MS" --logfile "$logfile" >/dev/null 2>&1 &
    TG_PID=$!
}

stop_tegrastats() {
    if [[ -n "${TG_PID:-}" ]]; then
        kill "$TG_PID" 2>/dev/null || true
        wait "$TG_PID" 2>/dev/null || true
        TG_PID=""
    fi
}

avg_ram_mb() {
    awk '
    match($0, /RAM [0-9]+\/[0-9]+MB/) {
        s=substr($0,RSTART,RLENGTH)
        sub("RAM ","",s)
        split(s,a,"/")
        sum+=a[1]
        n++
    }
    END {
        if (n > 0) printf "%.1f", sum/n
        else print "NA"
    }' "$1"
}

peak_ram_mb() {
    awk '
    match($0, /RAM [0-9]+\/[0-9]+MB/) {
        s=substr($0,RSTART,RLENGTH)
        sub("RAM ","",s)
        split(s,a,"/")
        ram=a[1]+0
        if (ram > max) max=ram
        n++
    }
    END {
        if (n > 0) printf "%.1f", max
        else print "NA"
    }' "$1"
}

delta_mb() {
    awk -v peak="$1" -v base="$2" 'BEGIN {
        if (peak=="NA" || base=="NA") print "NA"
        else printf "%.1f", peak-base
    }'
}

docker_base=(
    docker run --rm
    --runtime nvidia
    --network host
    --ipc=host
    --ulimit memlock=-1
    -v "$ROOT:/workspace"
    -w /workspace
    "$IMAGE"
)

run_case() {
    local backend="$1"
    local method="$2"
    local steps="${3:-}"
    local tag="${backend}_${method}"
    [[ -n "$steps" ]] && tag="${tag}_${steps}"

    local idle_log="$LOGDIR/${tag}_idle.log"
    local run_log="$LOGDIR/${tag}_run.log"
    local bench_out="/workspace/memory_benchmark/${tag}_benchmark.csv"

    echo
    echo "========================================"
    echo "Testing: $backend / $method ${steps:+/ steps=$steps}"
    echo "========================================"

    # Baseline RAM immediately before this test.
    start_tegrastats "$idle_log"
    sleep 1
    stop_tegrastats
    local baseline
    baseline="$(avg_ram_mb "$idle_log")"

    # Monitor the whole benchmark process.
    start_tegrastats "$run_log"
    sleep 0.2

    if [[ "$backend" == "pytorch" ]]; then
        cmd=(
            "${docker_base[@]}"
            python3 "$PYTORCH_SCRIPT"
            --device cuda
            --methods "$method"
            --warmup "$WARMUP"
            --iterations "$ITERATIONS"
            --output "$bench_out"
        )
    else
        cmd=(
            "${docker_base[@]}"
            python3 "$TRT_SCRIPT"
            --methods "$method"
            --warmup "$WARMUP"
            --iterations "$ITERATIONS"
            --output "$bench_out"
        )
    fi

    if [[ -n "$steps" ]]; then
        cmd+=(--ode-steps "$steps")
    fi

    "${cmd[@]}"

    sleep 0.2
    stop_tegrastats

    local peak delta
    peak="$(peak_ram_mb "$run_log")"
    delta="$(delta_mb "$peak" "$baseline")"

    echo "$backend,$method,${steps:-},$baseline,$peak,$delta" >> "$RESULT"

    echo "Baseline RAM : $baseline MB"
    echo "Peak RAM     : $peak MB"
    echo "Delta RAM    : $delta MB"

    sleep "$COOLDOWN_SEC"
}

# PyTorch
run_case pytorch cnn
for s in 10 20 50 100; do run_case pytorch gvc "$s"; done
for s in 10 20 50 100; do run_case pytorch dp  "$s"; done

# TensorRT
run_case tensorrt cnn
for s in 10 20 50 100; do run_case tensorrt gvc "$s"; done
for s in 10 20 50 100; do run_case tensorrt dp  "$s"; done

echo
echo "========================================"
echo "DONE"
echo "Results: $RESULT"
echo "Logs   : $LOGDIR"
echo "========================================"

if command -v column >/dev/null 2>&1; then
    echo
    column -s, -t "$RESULT"
else
    cat "$RESULT"
fi
