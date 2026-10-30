"""
GPU 持续计算守护进程，防止因 GPU 利用率低被集群 kill。

用法:
    python3 gpu_continuous_compute.py --gpu_number 8 --duration 0
    duration=0 表示永久运行，直到被 kill 为止。
"""
import os
import sys
import time
import argparse
import threading
import subprocess

import torch


def get_visible_gpu_indices(gpu_number: int):
    """
    返回当前进程实际可见的物理 GPU 编号列表。
    集群通过 CUDA_VISIBLE_DEVICES 做隔离，必须用这个才能查对应的物理卡。
    若未设置 CUDA_VISIBLE_DEVICES，则返回 [0, 1, ..., gpu_number-1]。
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd and cvd != "NoDevFiles":
        ids = [int(x.strip()) for x in cvd.split(",") if x.strip().isdigit()]
        return ids[:gpu_number]
    n = torch.cuda.device_count()
    return list(range(min(n, gpu_number)))


def gpu_utilization(physical_gpu_id: int) -> int:
    """用 nvidia-smi 查询指定物理 GPU 的当前利用率（%）。"""
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
    """
    每张卡独立一个线程，持续监控 + 矩阵乘法补偿。
    - 利用率 < threshold 时立即开始矩阵乘法
    - 利用率 >= threshold 时休眠 0.5s 后继续检测
    """
    device = torch.device(f"cuda:{logical_i}")
    sz = 8192  # A100 80GB: 8192x8192 float16 ~0.5 GB，能有效拉高利用率
    a = torch.rand(sz, sz, dtype=torch.float16, device=device)
    b = torch.rand(sz, sz, dtype=torch.float16, device=device)

    print(
        f"[gpu_keeper] 线程启动: logical={logical_i} physical={phys_i} threshold={threshold}%",
        flush=True,
    )

    while not stop_event.is_set():
        util = gpu_utilization(phys_i)
        if util < threshold:
            count = 0
            while not stop_event.is_set():
                for _ in range(10):
                    a = a @ b
                count += 1
                if count % 3 == 0:
                    torch.cuda.synchronize(logical_i)
                    cur_util = gpu_utilization(phys_i)
                    if cur_util >= threshold:
                        print(
                            f"[gpu_keeper] GPU{phys_i} 利用率恢复: {cur_util}% >= {threshold}%",
                            flush=True,
                        )
                        break
        else:
            time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description="GPU 持续计算守护进程")
    parser.add_argument("--gpu_number", type=int, default=1, help="监控的 GPU 数量")
    parser.add_argument(
        "--gpu_utilization_threshold",
        type=int,
        default=85,
        help="GPU 利用率阈值（低于此值则触发矩阵乘法补偿）",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0,
        help="运行时长（秒），0 表示永久运行",
    )
    args = parser.parse_args()

    physical_ids = get_visible_gpu_indices(args.gpu_number)
    print(
        f"[gpu_keeper] 监控物理 GPU: {physical_ids}, 阈值={args.gpu_utilization_threshold}%,"
        f" duration={'永久' if args.duration == 0 else f'{args.duration}s'}",
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
            # 永久运行，直到被 kill
            while True:
                time.sleep(60)
    except KeyboardInterrupt:
        print("[gpu_keeper] 收到中断，停止所有守护线程", flush=True)
        stop_event.set()

    for t in threads:
        t.join(timeout=3)


if __name__ == "__main__":
    main()
