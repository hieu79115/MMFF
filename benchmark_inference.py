"""Benchmark MMFF inference latency and GPU memory usage."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import Config
from models.mmff_net import MMFF_Net_Advanced
from utils.dataset import MMFFDataset


def ablation_tag(fusion_type: str, cross_attention: str) -> str:
    return f"{fusion_type}_attn-{cross_attention}"


def checkpoint_candidates(
    stage: str,
    dataset: str,
    fusion_type: str,
    cross_attention: str,
    seed: int | None,
) -> list[str]:
    seed_tag = f"_seed{seed}" if seed is not None else ""
    candidates = []
    if stage == "fusion":
        candidates.append(
            f"best_{stage}_{dataset}_{ablation_tag(fusion_type, cross_attention)}{seed_tag}.pth"
        )
    else:
        candidates.extend(
            [
                f"best_{stage}_{dataset}_attn-{cross_attention}{seed_tag}.pth",
                f"best_{stage}_{dataset}_{ablation_tag(fusion_type, cross_attention)}{seed_tag}.pth",
            ]
        )
    candidates.append(f"best_{stage}_{dataset}{seed_tag}.pth")
    if seed is None:
        candidates.append(f"best_model_{dataset}.pth")
    return candidates


def first_existing(paths: list[str]) -> str | None:
    return next((path for path in paths if os.path.exists(path)), None)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_forward(
    model: torch.nn.Module,
    skel: torch.Tensor,
    rgb: torch.Tensor,
    stage: str,
    device: torch.device,
) -> tuple[float, float]:
    """Return (forward_ms, transfer_plus_forward_ms)."""
    if device.type == "cuda":
        transfer_start = torch.cuda.Event(enable_timing=True)
        transfer_end = torch.cuda.Event(enable_timing=True)
        forward_start = torch.cuda.Event(enable_timing=True)
        forward_end = torch.cuda.Event(enable_timing=True)

        transfer_start.record()
        skel = skel.to(device, non_blocking=True)
        rgb = rgb.to(device, non_blocking=True)
        forward_start.record()
        model(skel, rgb, stage=stage)
        forward_end.record()
        transfer_end.record()
        transfer_end.synchronize()
        return forward_start.elapsed_time(forward_end), transfer_start.elapsed_time(transfer_end)

    transfer_start = time.perf_counter()
    skel = skel.to(device)
    rgb = rgb.to(device)
    transfer_end = time.perf_counter()
    forward_start = time.perf_counter()
    model(skel, rgb, stage=stage)
    forward_end = time.perf_counter()
    return (
        (forward_end - forward_start) * 1000,
        (forward_end - transfer_start) * 1000,
    )


def run_benchmark(args: argparse.Namespace) -> dict:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    args.dataset = Config.normalize_dataset(args.dataset)
    num_classes = Config.get_num_classes(args.dataset)
    checkpoint = args.checkpoint or first_existing(
        checkpoint_candidates(
            args.stage,
            args.dataset,
            args.fusion_type,
            args.cross_attention,
            args.seed,
        )
    )
    if not checkpoint:
        raise FileNotFoundError("No checkpoint found. Pass --checkpoint or train a model first.")

    model = MMFF_Net_Advanced(
        num_classes=num_classes,
        dataset=args.dataset,
        edge_importance_weighting=bool(args.edge_importance),
        stgcn_dropout=args.dropout,
        fusion_type=args.fusion_type,
        cross_attention_mode=args.cross_attention,
    )
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    dataset = MMFFDataset(
        root_dir=args.data_dir,
        mode="test",
        is_dummy=args.is_dummy,
        num_samples=max(args.max_samples, args.batch_size),
        num_classes=num_classes,
        dataset=args.dataset,
        stage=args.stage,
        num_frames=args.num_frames,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    batches = []
    collected = 0
    for batch in loader:
        batches.append(batch)
        collected += batch[0].shape[0]
        if collected >= args.max_samples:
            break
    if not batches:
        raise RuntimeError("The benchmark dataset is empty.")

    with torch.inference_mode():
        for batch in batches[: args.warmup_batches]:
            timed_forward(model, batch[0], batch[1], args.stage, device)

    if device.type == "cuda":
        synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    forward_ms = 0.0
    transfer_forward_ms = 0.0
    total_samples = 0
    measured_batches = 0
    with torch.inference_mode():
        for batch in batches:
            remaining = args.max_samples - total_samples
            if remaining <= 0:
                break
            batch_size = min(batch[0].shape[0], remaining)
            skel = batch[0][:batch_size]
            rgb = batch[1][:batch_size]
            measured_forward_ms, measured_transfer_ms = timed_forward(
                model, skel, rgb, args.stage, device
            )
            forward_ms += measured_forward_ms
            transfer_forward_ms += measured_transfer_ms
            total_samples += batch_size
            measured_batches += 1

    if total_samples == 0:
        raise RuntimeError("No samples were measured.")

    peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
    result = {
        "device": str(device),
        "checkpoint": str(Path(checkpoint)),
        "dataset": args.dataset,
        "stage": args.stage,
        "batch_size": args.batch_size,
        "num_frames": args.num_frames,
        "samples": total_samples,
        "measured_batches": measured_batches,
        "warmup_batches": min(args.warmup_batches, len(batches)),
        "forward_total_ms": forward_ms,
        "transfer_plus_forward_total_ms": transfer_forward_ms,
        "forward_latency_ms_per_sample": forward_ms / total_samples,
        "forward_latency_ms_per_frame": forward_ms / total_samples / args.num_frames,
        "transfer_plus_forward_ms_per_sample": transfer_forward_ms / total_samples,
        "throughput_samples_per_second": total_samples / (forward_ms / 1000),
        "throughput_frames_per_second": total_samples * args.num_frames / (forward_ms / 1000),
        "peak_gpu_memory_allocated_mb": peak_allocated / 1024**2,
        "peak_gpu_memory_reserved_mb": peak_reserved / 1024**2,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset", choices=["ntu", "utd", "nw-ucla", "sumv2", "sgu-sb"], default="ntu")
    parser.add_argument("--stage", choices=["skeleton", "rgb", "fusion"], default="fusion")
    parser.add_argument("--fusion_type", choices=["cmaf", "sum", "average", "concat"], default="cmaf")
    parser.add_argument("--cross_attention", choices=["normal", "none", "reversed"], default="normal")
    parser.add_argument("--seed", type=int, default=None, help="Select a seed-specific checkpoint.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--warmup_batches", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--edge_importance", type=int, choices=[0, 1], default=0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--is_dummy", action="store_true")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    if args.max_samples < 1 or args.warmup_batches < 0 or args.batch_size < 1:
        parser.error("--max_samples, --batch_size must be positive and --warmup_batches cannot be negative.")
    result = run_benchmark(args)
    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
