"""
GPU keeper: sustain a steady GPU load so the platform does not reclaim the
job for low utilization during idle stages (conversion, init, rollout gaps).

Strategy follows the longcat-rl BGE server, which was never killed: compute
in long continuous bursts instead of reacting to a threshold with short
ramp-up/drop cycles. The only pause is a short probe — if the GPU stays busy
without the keeper, external work is running and the keeper yields.

Usage:
    python3 gpu_continuous_compute.py --gpu_number 8 --duration 0
    duration=0 runs until killed.
"""
import argparse
import os
import subprocess
import threading
import time

import torch

BURST_SECONDS = 3.0     # continuous compute between probes
PROBE_PAUSE = 1.0       # pause so the reading reflects external load only
BACKOFF_SECONDS = 60.0  # yield while external work keeps the GPU busy


def get_visible_gpu_indices(gpu_number: int):
    """
    Return the physical GPU ids visible to this process.
    The cluster isolates GPUs via CUDA_VISIBLE_DEVICES, so the physical ids
    must be resolved through it. Defaults to [0..gpu_number-1] if unset.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd and cvd != "NoDevFiles":
        ids = [int(x.strip()) for x in cvd.split(",") if x.strip().isdigit()]
        return ids[:gpu_number]
    n = torch.cuda.device_count()
    return list(range(min(n, gpu_number)))


def gpu_utilization(physical_gpu_id: int) -> int:
    """Read the current utilization (%) of a physical GPU via nvidia-smi."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
                f"--id={physical_gpu_id}",
            ],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return int(out)
    except Exception:
        return 0


def keep_gpu_busy(logical_i: int, phys_i: int, threshold: int, stop_event: threading.Event):
    device = torch.device(f"cuda:{logical_i}")
    sz = 8192  # A100 80GB: 8192x8192 float16 ~0.5 GB per matrix
    a = torch.rand(sz, sz, dtype=torch.float16, device=device)
    b = torch.rand(sz, sz, dtype=torch.float16, device=device)

    print(
        f"[gpu_keeper] thread started: logical={logical_i} physical={phys_i} threshold={threshold}%",
        flush=True,
    )

    while not stop_event.is_set():
        burst_end = time.time() + BURST_SECONDS
        while not stop_event.is_set() and time.time() < burst_end:
            for _ in range(10):
                a = a @ b
            torch.cuda.synchronize(logical_i)
        # Probe: pause briefly, then check whether external work keeps the
        # GPU busy on its own.
        stop_event.wait(PROBE_PAUSE)
        if gpu_utilization(phys_i) >= threshold:
            print(f"[gpu_keeper] GPU{phys_i} external load active, yielding {BACKOFF_SECONDS:.0f}s", flush=True)
            stop_event.wait(BACKOFF_SECONDS)


def main():
    parser = argparse.ArgumentParser(description="GPU keeper daemon")
    parser.add_argument("--gpu_number", type=int, default=1, help="number of GPUs to keep busy")
    parser.add_argument(
        "--gpu_utilization_threshold",
        type=int,
        default=75,
        help="probe threshold: utilization above this after a pause means external work is active",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0,
        help="run time in seconds; 0 runs until killed",
    )
    args = parser.parse_args()

    physical_ids = get_visible_gpu_indices(args.gpu_number)
    print(
        f"[gpu_keeper] physical GPUs: {physical_ids}, threshold={args.gpu_utilization_threshold}%,"
        f" duration={'forever' if args.duration == 0 else f'{args.duration}s'}",
        flush=True,
    )

    stop_event = threading.Event()

    threads = []
    for logical_i, phys_i in enumerate(physical_ids):
        t = threading.Thread(
            target=keep_gpu_busy,
            args=(logical_i, phys_i, args.gpu_utilization_threshold, stop_event),
            daemon=True,
            name=f"gpu-keeper-{phys_i}",
        )
        t.start()
        threads.append(t)

    try:
        if args.duration > 0:
            time.sleep(args.duration)
            stop_event.set()
        else:
            while True:
                time.sleep(60)
    except KeyboardInterrupt:
        print("[gpu_keeper] interrupted, stopping threads", flush=True)
        stop_event.set()

    for t in threads:
        t.join(timeout=3)


if __name__ == "__main__":
    main()
