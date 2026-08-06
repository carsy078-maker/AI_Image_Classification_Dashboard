"""사용 가능한 모든 변형을 용량 / 지연시간 / 정확도로 비교한다.

    python scripts/run_benchmark.py --n 200 --runs 30

결과는 results/benchmark.json 과 results/benchmark.md 로 저장되고,
대시보드의 벤치마크 탭이 같은 파일을 읽는다.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vision_dashboard.backends import available_keys  # noqa: E402
from vision_dashboard.benchmarking import (  # noqa: E402
    RESULTS_PATH,
    comparison_rows,
    load_samples,
    run_benchmark,
    save_results,
)
from vision_dashboard.config import VARIANTS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="ViT 변형별 벤치마크")
    parser.add_argument("--n", type=int, default=200, help="평가에 쓸 이미지 수")
    parser.add_argument("--runs", type=int, default=30, help="지연시간 측정 반복 횟수")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4, help="PyTorch/ONNX 공통 스레드 수")
    parser.add_argument(
        "--only", nargs="+", choices=list(VARIANTS), help="측정할 변형 (기본: 사용 가능한 전부)"
    )
    args = parser.parse_args()

    # config.cpu_threads() 가 이 값을 읽는다. 두 런타임에 같은 값이 들어가야
    # '런타임 비교' 가 '스레드 수 비교' 로 변질되지 않는다.
    os.environ["VD_CPU_THREADS"] = str(args.threads)

    keys = args.only or available_keys()
    if not keys:
        raise SystemExit("사용 가능한 변형이 없다. scripts/export_models.py 를 먼저 실행할 것.")
    print(f"측정 대상: {', '.join(keys)}")

    print(f"샘플 이미지 로드 (최대 {args.n}장)...")
    inputs, labels = load_samples(args.n)
    print(f"  {len(inputs)}장 준비 완료")

    payload = run_benchmark(
        keys,
        inputs,
        labels,
        runs=args.runs,
        warmup=args.warmup,
        batch_size=args.batch_size,
        on_progress=lambda _key, msg: print(f"  {msg}"),
    )
    save_results(payload)

    print()
    for row in comparison_rows(payload):
        print(
            f"  {row['변형']:<24} {str(row['용량(MB)']):>8} MB  "
            f"{row['지연 mean(ms)']:>7.1f} ms  {row['속도']:>5.2f}x  "
            f"top1 {row['Top-1(%)']:>5.1f}%  FP32일치 {row['FP32 일치(%)']:>5.1f}%"
        )
    print(f"\n결과 저장 -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
