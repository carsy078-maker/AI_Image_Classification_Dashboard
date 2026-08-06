"""벤치마크 집계 로직.

실측은 시간이 오래 걸리므로 계산 부분만 떼어 검증한다.
"""

import json

import numpy as np
import pytest

from vision_dashboard import benchmarking


def test_accuracy_counts_top1_and_top5():
    # 클래스가 5개 이하면 top-5 가 전체 집합이 되어 아무것도 변별하지 못한다.
    logits = np.array(
        [
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],  # 정답 0 -> 1등, top1 O
            [0, 1, 2, 3, 8, 4, 3, 2, 1, 9],  # 정답 4 -> 2등, top1 X / top5 O
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],  # 정답 9 -> 꼴등, 둘 다 X
        ],
        dtype=np.float32,
    )
    acc = benchmarking.accuracy(logits, np.array([0, 4, 9]))
    assert acc["top1"] == pytest.approx(33.33, abs=0.01)
    assert acc["top5"] == pytest.approx(66.67, abs=0.01)


def test_measure_latency_reports_ordered_percentiles():
    stats = benchmarking.measure_latency(lambda x: x, np.zeros((1, 3, 4, 4)), runs=10, warmup=2)
    assert stats["runs"] == 10
    assert stats["p50_ms"] <= stats["p95_ms"]
    assert stats["mean_ms"] >= 0


def _payload() -> dict:
    def entry(label, size, mean, p95, top1, agree):
        return {
            "label": label,
            "short": label,
            "runtime": "onnx",
            "precision": "int8",
            "file": "x",
            "size_mb": size,
            "latency": {"mean_ms": mean, "p50_ms": mean, "p95_ms": p95, "runs": 30},
            "accuracy": {"top1": top1, "top5": 99.0},
            "fp32_agreement_pct": agree,
            "throughput_img_per_sec": 10.0,
        }

    return {
        "model": "google/vit-base-patch16-224",
        "num_eval_images": 200,
        "batch_size": 8,
        "environment": {
            "python": "3.12.0",
            "platform": "test",
            "processor": "test",
            "cpu_threads": 4,
        },
        "variants": {
            "pytorch_fp32": entry("PyTorch FP32", 330.0, 40.0, 42.0, 88.0, 100.0),
            "onnx_int8": entry("ONNX INT8", 82.5, 20.0, 21.0, 87.5, 97.5),
        },
    }


def test_comparison_rows_computes_relative_metrics():
    rows = {r["key"]: r for r in benchmarking.comparison_rows(_payload())}
    onnx = rows["onnx_int8"]
    assert onnx["용량 절감(%)"] == pytest.approx(75.0)
    assert onnx["속도"] == pytest.approx(2.0)


def test_baseline_row_is_neutral():
    base = {r["key"]: r for r in benchmarking.comparison_rows(_payload())}["pytorch_fp32"]
    assert base["속도"] == pytest.approx(1.0)
    assert base["용량 절감(%)"] == pytest.approx(0.0)


def test_comparison_rows_tolerate_missing_size():
    """Hub 에서 받아 쓰는 변형은 로컬 파일이 없어 용량을 모를 수 있다."""
    payload = _payload()
    payload["variants"]["onnx_int8"]["size_mb"] = None
    row = {r["key"]: r for r in benchmarking.comparison_rows(payload)}["onnx_int8"]
    assert row["용량(MB)"] is None
    assert row["용량 절감(%)"] is None


def test_comparison_rows_without_pytorch_baseline():
    """PyTorch 를 못 쓰는 배포 환경에서도 첫 항목을 기준선 삼아 동작해야 한다."""
    payload = _payload()
    del payload["variants"]["pytorch_fp32"]
    rows = benchmarking.comparison_rows(payload)
    assert len(rows) == 1
    assert rows[0]["속도"] == pytest.approx(1.0)


def test_write_markdown_renders_every_variant(tmp_path):
    path = benchmarking.write_markdown(_payload(), tmp_path / "b.md")
    text = path.read_text(encoding="utf-8")
    assert "PyTorch FP32" in text and "ONNX INT8" in text
    assert "75.0%" in text


def test_load_results_returns_none_when_absent(tmp_path):
    assert benchmarking.load_results(tmp_path / "missing.json") is None


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmarking, "RESULTS_DIR", tmp_path)
    path = tmp_path / "benchmark.json"
    payload = _payload()
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    assert benchmarking.load_results(path) == payload


def test_load_samples_reports_missing_index(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmarking, "SAMPLES_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="download_samples"):
        benchmarking.load_samples(10)
