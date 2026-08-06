"""이미지 로드 · 전처리 · 배치 추론 · 후처리.

모델 가중치 없이 돌도록 백엔드는 가짜로 대체한다. 여기서 검증하는 건
추론 파이프라인의 배선이지 모델의 정확도가 아니다.
"""

import io

import numpy as np
import pytest
from PIL import Image

from vision_dashboard import inference
from vision_dashboard.inference import (
    ImageLoadError,
    ImageResult,
    classify,
    load_image,
    softmax,
    thumbnail,
)


class FakeBackend:
    """호출 횟수와 배치 크기를 기록하는 가짜 백엔드."""

    key = "fake"
    label = "Fake"

    def __init__(self):
        self.calls = 0
        self.batch_sizes = []

    def infer(self, pixel_values: np.ndarray) -> np.ndarray:
        self.calls += 1
        self.batch_sizes.append(len(pixel_values))
        n = len(pixel_values)
        logits = np.zeros((n, 1000), dtype=np.float32)
        # 이미지마다 다른 클래스가 1등이 되게 해 결과 대응이 섞이지 않는지 본다.
        for i in range(n):
            logits[i, (i * 7) % 1000] = 10.0
        return logits


# --- 이미지 로드 ---


def test_load_image_returns_rgb(sample_image_path):
    img = load_image(sample_image_path)
    assert img.mode == "RGB"


def test_load_image_converts_rgba(tmp_path):
    path = tmp_path / "rgba.png"
    Image.new("RGBA", (32, 32), (255, 0, 0, 128)).save(path)
    assert load_image(path).mode == "RGB"


def test_load_image_converts_grayscale(tmp_path):
    path = tmp_path / "gray.png"
    Image.new("L", (32, 32), 128).save(path)
    assert load_image(path).mode == "RGB"


def test_load_image_raises_on_corrupt_file(tmp_path):
    """손상된 파일 하나가 앱 전체를 죽이면 안 된다. 잡을 수 있는 예외여야 한다."""
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"not an image at all")
    with pytest.raises(ImageLoadError):
        load_image(path, "broken.jpg")


def test_load_image_error_message_includes_name(tmp_path):
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"\x00\x01\x02")
    with pytest.raises(ImageLoadError, match="broken.jpg"):
        load_image(path, "broken.jpg")


def test_load_image_accepts_file_like(sample_image_bytes):
    assert load_image(io.BytesIO(sample_image_bytes)).mode == "RGB"


# --- 썸네일 ---


def test_thumbnail_shrinks_large_image():
    big = Image.new("RGB", (4000, 3000))
    small = thumbnail(big, max_pixels=100_000)
    assert small.size[0] * small.size[1] <= 100_000
    # 가로세로 비율은 유지해야 한다
    assert abs(small.size[0] / small.size[1] - 4000 / 3000) < 0.05


def test_thumbnail_leaves_small_image_untouched():
    small = Image.new("RGB", (100, 100))
    assert thumbnail(small, max_pixels=100_000) is small


# --- 수치 ---


def test_softmax_rows_sum_to_one():
    probs = softmax(np.random.randn(4, 1000))
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_softmax_handles_1d_input():
    assert softmax(np.random.randn(1000)).shape == (1, 1000)


def test_softmax_is_stable_for_large_logits():
    """max 를 빼지 않으면 exp 가 overflow 해 nan 이 된다."""
    probs = softmax(np.array([[1000.0, 1001.0, 999.0]]))
    assert np.isfinite(probs).all()
    assert np.allclose(probs.sum(), 1.0)


# --- ImageResult ---


def _result(probs=None) -> ImageResult:
    if probs is None:
        probs = np.zeros(1000, dtype=np.float64)
        probs[5], probs[10], probs[20] = 0.5, 0.3, 0.2
    return ImageResult(name="x", probs=probs, latency_ms=1.0)


def test_top_k_is_sorted_descending():
    scores = [p.score for p in _result().top(3)]
    assert scores == sorted(scores, reverse=True)


def test_top_k_returns_correct_indices():
    top = _result().top(3)
    assert [p.index for p in top] == [5, 10, 20]


def test_top_clamps_out_of_range_k():
    assert len(_result().top(0)) == 1
    assert len(_result().top(5000)) == 1000


def test_best_is_the_highest_scoring_class():
    assert _result().best.index == 5


def test_top_k_change_does_not_need_new_inference():
    """확률 전체를 들고 있으므로 K 를 바꿔도 재추론이 필요 없다.

    원본 앱은 슬라이더를 움직일 때마다 모델을 다시 돌렸다.
    """
    res = _result()
    assert [p.index for p in res.top(1)] == [5]
    assert [p.index for p in res.top(3)] == [5, 10, 20]


# --- 배치 추론 ---


def test_classify_returns_one_result_per_image(sample_image_path):
    img = load_image(sample_image_path)
    results = classify(FakeBackend(), [img] * 3, ["a", "b", "c"])
    assert [r.name for r in results] == ["a", "b", "c"]


def test_classify_batches_instead_of_looping_per_image(sample_image_path):
    """5장을 batch_size=2 로 넣으면 백엔드 호출은 3번이어야 한다 (5번이 아니라)."""
    img = load_image(sample_image_path)
    backend = FakeBackend()
    classify(backend, [img] * 5, batch_size=2)
    assert backend.calls == 3
    assert backend.batch_sizes == [2, 2, 1]


def test_classify_keeps_image_order(sample_image_path):
    """가짜 백엔드는 i 번째 이미지의 1등을 (i*7) 로 만든다."""
    img = load_image(sample_image_path)
    results = classify(FakeBackend(), [img] * 4, batch_size=4)
    assert [r.best.index for r in results] == [0, 7, 14, 21]


def test_classify_reports_batch_progress(sample_image_path):
    """UI 가 청크를 다시 쪼개지 않고도 진행률을 그릴 수 있어야 한다."""
    img = load_image(sample_image_path)
    seen: list[tuple[int, int]] = []
    classify(FakeBackend(), [img] * 5, batch_size=2, on_batch=lambda d, t: seen.append((d, t)))
    assert seen == [(2, 5), (4, 5), (5, 5)]


def test_classify_without_callback_still_works(sample_image_path):
    img = load_image(sample_image_path)
    assert len(classify(FakeBackend(), [img] * 3, batch_size=2)) == 3


def test_batch_callback_time_excluded_from_latency(sample_image_path):
    """콜백에서 UI 를 그리는 시간이 추론 지연시간으로 집계되면 안 된다."""
    import time as _time

    img = load_image(sample_image_path)
    res = classify(FakeBackend(), [img] * 2, batch_size=1, on_batch=lambda d, t: _time.sleep(0.05))
    # 콜백이 총 0.1초를 먹었다. 그게 집계되면 이미지당 50ms 를 넘는다.
    assert res[0].latency_ms < 40


def test_classify_generates_default_names(sample_image_path):
    img = load_image(sample_image_path)
    results = classify(FakeBackend(), [img, img])
    assert [r.name for r in results] == ["image_1", "image_2"]


def test_classify_probs_are_normalized(sample_image_path):
    img = load_image(sample_image_path)
    res = classify(FakeBackend(), [img])[0]
    assert res.probs.shape == (1000,)
    assert res.probs.sum() == pytest.approx(1.0)


def test_preprocess_rejects_empty_input():
    with pytest.raises(ValueError):
        inference.preprocess([])
