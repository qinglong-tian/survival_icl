"""Validate Fir CUDA execution and optional single-node NCCL communication."""

from __future__ import annotations

import argparse
import ctypes
import os
from datetime import timedelta

import torch
import torch.distributed as dist


def _cuda_error_text(driver: ctypes.CDLL, result: int) -> str:
    parts = [f"code {result}"]
    for function_name in ("cuGetErrorName", "cuGetErrorString"):
        function = getattr(driver, function_name)
        function.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        function.restype = ctypes.c_int
        value = ctypes.c_char_p()
        if function(result, ctypes.byref(value)) == 0 and value.value is not None:
            parts.append(value.value.decode())
    return ": ".join(parts)


def initialize_driver() -> None:
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuInit.restype = ctypes.c_int
    result = driver.cuInit(0)
    if result != 0:
        raise RuntimeError(f"CUDA driver cuInit failed with {_cuda_error_text(driver, result)}.")


def check_local_cuda(expected_gpus: int) -> None:
    initialize_driver()
    torch.cuda.init()
    actual_gpus = torch.cuda.device_count()
    if actual_gpus != expected_gpus:
        raise RuntimeError(f"Expected {expected_gpus} CUDA devices, found {actual_gpus}.")

    tensors = [torch.ones(1, device=f"cuda:{index}") for index in range(actual_gpus)]
    for tensor in tensors:
        tensor.add_(1)
    torch.cuda.synchronize()
    print(f"CUDA execution preflight: OK on {actual_gpus} GPUs")


def check_distributed_cuda(expected_gpus: int) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != expected_gpus:
        raise RuntimeError(f"Expected world size {expected_gpus}, found {world_size}.")

    initialize_driver()
    torch.cuda.set_device(local_rank)
    value = torch.tensor([local_rank + 1.0], device=f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=60))
    try:
        dist.all_reduce(value)
        expected_sum = world_size * (world_size + 1) / 2
        if value.item() != expected_sum:
            raise RuntimeError(f"NCCL all-reduce returned {value.item()}, expected {expected_sum}.")
        if local_rank == 0:
            print(f"NCCL preflight: OK across {world_size} ranks")
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-gpus", type=int, required=True)
    parser.add_argument("--distributed", action="store_true")
    args = parser.parse_args()

    print(
        f"PyTorch {torch.__version__}; CUDA runtime {torch.version.cuda}; "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}"
    )
    if args.distributed:
        check_distributed_cuda(args.expected_gpus)
    else:
        check_local_cuda(args.expected_gpus)


if __name__ == "__main__":
    main()
