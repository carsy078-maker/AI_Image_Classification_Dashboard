"""변형별 용량 / 지연시간 / 정확도 측정.

CLI(`scripts/run_benchmark.py`)와 대시보드의 벤치마크 탭이 같은 함수를 쓴다.
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
from PIL import Image

from .backends import load_backend, variant_status
from .config import MODEL_NAME, RESULTS_DIR, SAMPLES_DIR, VARIANTS, cpu_threads, ensure_dirs
from .inference import get_processor, infer_batched

RESULTS_PATH = RESULTS_DIR / "benchmark.json"


def load_samples(limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """전처리된 입력 배열과 정답 라벨을 돌려준다."""
    index_path = SAMPLES_DIR / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            "샘플 이미지가 없다. `python scripts/download_samples.py --n 200` 을 먼저 실행할 것."
        )
    records = json.loads(index_path.read_text(encoding="utf-8"))
    if limit is not None:
        records = records[:limit]

    processor = get_processor()
    images, labels = [], []
    for rec in records:
        path = SAMPLES_DIR / rec["path"]
        if not path.exists():
            continue
        with Image.open(path) as img:
            images.append(img.convert("RGB").copy())
        labels.append(rec["label"])
    if not images:
        raise FileNotFoundError("읽을 수 있는 샘플 이미지가 없다.")

    tensors = processor(images=images, return_tensors="np")["pixel_values"]
    return tensors.astype(np.float32), np.array(labels)


def measure_latency(
    infer: Callable[[np.ndarray], object], sample: np.ndarray, runs: int, warmup: int
) -> dict:
    """단일 이미지 추론을 warmup 후 runs 회 반복해 지연시간 통계를 낸다."""
    for _ in range(warmup):
        infer(sample)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        infer(sample)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {
        "mean_ms": round(statistics.fmean(times), 2),
        "p50_ms": round(times[len(times) // 2], 2),
        "p95_ms": round(times[min(len(times) - 1, int(len(times) * 0.95))], 2),
        "runs": runs,
    }


def accuracy(logits: np.ndarray, labels: np.ndarray) -> dict:
    top1 = logits.argmax(axis=1)
    # argsort 는 오름차순이므로 뒤에서 5개가 상위 5개다.
    top5 = np.argsort(logits, axis=1)[:, -5:]
    return {
        "top1": round(float((top1 == labels).mean()) * 100, 2),
        "top5": round(float((top5 == labels[:, None]).any(axis=1).mean()) * 100, 2),
    }


def run_benchmark(
    keys: Iterable[str],
    inputs: np.ndarray,
    labels: np.ndarray,
    runs: int = 30,
    warmup: int = 5,
    batch_size: int = 8,
    on_progress: Callable[[str, str], None] | None = None,
) -> dict:
    """지정한 변형들을 순서대로 측정한다.

    첫 번째로 측정한 변형이 FP32 일치율의 기준선이 된다. PyTorch FP32 를
    포함시키면 그것이 기준선이 되도록 VARIANTS 순서를 따른다.
    """
    ordered = [k for k in VARIANTS if k in set(keys)]
    if not ordered:
        raise ValueError("측정할 변형이 없다")

    sample = inputs[:1]
    results: dict[str, dict] = {}
    baseline_top1: np.ndarray | None = None

    for key in ordered:
        meta = VARIANTS[key]
        if on_progress:
            on_progress(key, f"{meta['label']} 로드 중")
        backend = load_backend(key)

        if on_progress:
            on_progress(key, f"{meta['label']} 지연시간 측정 중")
        latency = measure_latency(backend.infer, sample, runs, warmup)

        if on_progress:
            on_progress(key, f"{meta['label']} 정확도 측정 중 ({len(inputs)}장)")
        t0 = time.perf_counter()
        logits, _ = infer_batched(backend, inputs, batch_size)
        eval_sec = time.perf_counter() - t0

        acc = accuracy(logits, labels)
        top1_pred = logits.argmax(axis=1)
        if baseline_top1 is None:
            baseline_top1 = top1_pred
        agreement = round(float((top1_pred == baseline_top1).mean()) * 100, 2)

        status = variant_status(key)
        results[key] = {
            "label": meta["label"],
            "short": meta["short"],
            "runtime": meta["runtime"],
            "precision": meta["precision"],
            "file": meta["path"].name,
            "size_mb": status.size_mb,
            "latency": latency,
            "accuracy": acc,
            "fp32_agreement_pct": agreement,
            "throughput_img_per_sec": round(len(inputs) / eval_sec, 2),
        }

    return {
        "model": MODEL_NAME,
        "num_eval_images": int(len(inputs)),
        "batch_size": batch_size,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "cpu_threads": cpu_threads(),
        },
        "variants": results,
    }


def save_results(payload: dict, path: Path = RESULTS_PATH) -> None:
    ensure_dirs()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(payload)


def load_results(path: Path = RESULTS_PATH) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def comparison_rows(payload: dict) -> list[dict]:
    """기준선 대비 상대 지표를 붙인 표 데이터.

    기준선은 PyTorch FP32 로 하되, 측정 대상에 없으면 첫 항목을 쓴다.
    """
    variants = payload["variants"]
    base_key = "pytorch_fp32" if "pytorch_fp32" in variants else next(iter(variants))
    base = variants[base_key]

    rows = []
    for key, v in variants.items():
        size = v.get("size_mb")
        base_size = base.get("size_mb")
        rows.append(
            {
                "key": key,
                "변형": v["label"],
                "용량(MB)": size,
                "용량 절감(%)": (
                    round((1 - size / base_size) * 100, 1) if size and base_size else None
                ),
                "지연 mean(ms)": v["latency"]["mean_ms"],
                "지연 p95(ms)": v["latency"]["p95_ms"],
                "속도": round(base["latency"]["mean_ms"] / v["latency"]["mean_ms"], 2),
                "Top-1(%)": v["accuracy"]["top1"],
                "Top-5(%)": v["accuracy"]["top5"],
                "FP32 일치(%)": v["fp32_agreement_pct"],
            }
        )
    return rows


def write_markdown(payload: dict, path: Path | None = None) -> Path:
    path = path or (RESULTS_DIR / "benchmark.md")
    env = payload["environment"]
    lines = [
        "# 벤치마크 결과",
        "",
        f"- 모델: `{payload['model']}`",
        f"- 평가 이미지: {payload['num_eval_images']}장 (ImageNet-1k 에서 균등 추출)",
        f"- 환경: {env['platform']}, Python {env['python']}, "
        f"CPU {env['cpu_threads']} threads (PyTorch/ONNX 동일)",
        "",
        "| 변형 | 용량 | 용량 절감 | 지연 mean | 지연 p95 | 속도 | Top-1 | Top-5 | FP32 일치 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_rows(payload):
        size = f"{row['용량(MB)']:.1f} MB" if row["용량(MB)"] else "-"
        cut = f"{row['용량 절감(%)']:.1f}%" if row["용량 절감(%)"] is not None else "-"
        lines.append(
            f"| {row['변형']} | {size} | {cut} | {row['지연 mean(ms)']:.1f} ms | "
            f"{row['지연 p95(ms)']:.1f} ms | {row['속도']:.2f}x | {row['Top-1(%)']:.1f}% | "
            f"{row['Top-5(%)']:.1f}% | {row['FP32 일치(%)']:.1f}% |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
