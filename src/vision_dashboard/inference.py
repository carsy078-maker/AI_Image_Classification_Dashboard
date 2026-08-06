"""이미지 로드 · 전처리 · 배치 추론 · 후처리."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .backends import Backend
from .config import INFER_BATCH_SIZE, MAX_PREVIEW_PIXELS, MODEL_NAME
from .labels import label_of


class ImageLoadError(Exception):
    """이미지를 열 수 없을 때. 앱 전체를 죽이지 않고 해당 파일만 건너뛰기 위해 쓴다."""


@dataclass(frozen=True)
class Prediction:
    index: int
    label: str
    score: float


@dataclass
class ImageResult:
    """이미지 1장에 대한 한 백엔드의 추론 결과.

    확률 1000개를 통째로 들고 있는 게 핵심이다. Top-K 개수를 바꾼다고
    추론을 다시 돌릴 이유가 없다.
    """

    name: str
    probs: np.ndarray
    latency_ms: float

    def top(self, k: int) -> list[Prediction]:
        k = max(1, min(k, self.probs.shape[-1]))
        idx = np.argpartition(-self.probs, k - 1)[:k]
        idx = idx[np.argsort(-self.probs[idx])]
        return [Prediction(int(i), label_of(int(i)), float(self.probs[i])) for i in idx]

    @property
    def best(self) -> Prediction:
        return self.top(1)[0]


@lru_cache(maxsize=1)
def get_processor():
    """이미지 전처리기. PIL 백엔드로 고정한다.

    transformers 는 torchvision 이 있으면 그쪽 백엔드를 쓰는데, 리사이즈 구현이
    PIL 백엔드와 미세하게 다르다(같은 이미지에서 최대 0.008 차이). 개발 환경엔
    torch 가 있고 배포 환경엔 없으므로, 백엔드를 고정하지 않으면 같은 모델이
    환경에 따라 다른 입력을 받게 되고 벤치마크 수치의 재현성이 깨진다.

    PIL 쪽으로 통일하면 torch 없이도 ONNX 백엔드만으로 배포할 수 있다는
    이점도 따라온다.
    """
    try:
        from transformers import ViTImageProcessorPil

        return ViTImageProcessorPil.from_pretrained(MODEL_NAME)
    except ImportError:  # transformers 4.x 대비
        from transformers import AutoImageProcessor

        return AutoImageProcessor.from_pretrained(MODEL_NAME)


def load_image(source: BinaryIO | str, name: str = "image") -> Image.Image:
    """업로드 파일을 PIL 이미지로 연다.

    EXIF 회전 정보를 반영하고 RGB 로 통일한다. 휴대폰 사진은 세로로 찍어도
    바이트는 가로로 저장되고 방향은 EXIF 태그에만 들어있어, 보정하지 않으면
    눕혀진 이미지를 추론하게 된다.
    """
    try:
        img = Image.open(source)
        img.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageLoadError(f"{name}: 이미지를 읽을 수 없다 ({exc})") from exc

    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def thumbnail(image: Image.Image, max_pixels: int = MAX_PREVIEW_PIXELS) -> Image.Image:
    """화면 표시용 축소본. 원본 해상도를 그대로 브라우저로 보내지 않기 위한 것."""
    w, h = image.size
    if w * h <= max_pixels:
        return image
    scale = (max_pixels / (w * h)) ** 0.5
    return image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


def preprocess(images: list[Image.Image]) -> np.ndarray:
    """PIL 이미지 목록 -> (N, 3, 224, 224) float32."""
    if not images:
        raise ValueError("전처리할 이미지가 없다")
    out = get_processor()(images=images, return_tensors="np")
    return out["pixel_values"].astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    """행 단위 softmax. 배치가 1보다 커도 올바르게 동작한다."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def infer_batched(
    backend: Backend,
    pixel_values: np.ndarray,
    batch_size: int = INFER_BATCH_SIZE,
    on_batch: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, float]:
    """배치로 나눠 추론한다. (logits, 총 소요 ms) 를 돌려준다.

    ONNX 그래프는 배치 축이 동적으로 열려 있고 PyTorch 도 배치 입력을 받으므로,
    이미지를 한 장씩 넣을 이유가 없다.

    on_batch 는 배치가 끝날 때마다 (처리한 장수, 전체 장수) 로 호출한다.
    진행률을 보여주려고 호출한 쪽에서 청크를 다시 쪼갤 필요가 없게 하려는 것이다.
    콜백에 걸린 시간은 지연시간 집계에서 제외한다.
    """
    chunks = []
    total = len(pixel_values)
    elapsed_ms = 0.0
    for start in range(0, total, batch_size):
        t0 = time.perf_counter()
        chunks.append(backend.infer(pixel_values[start : start + batch_size]))
        elapsed_ms += (time.perf_counter() - t0) * 1000.0
        if on_batch:
            on_batch(min(start + batch_size, total), total)
    return np.concatenate(chunks, axis=0), elapsed_ms


def classify(
    backend: Backend,
    images: list[Image.Image],
    names: list[str] | None = None,
    batch_size: int = INFER_BATCH_SIZE,
    on_batch: Callable[[int, int], None] | None = None,
) -> list[ImageResult]:
    """이미지 목록을 분류한다. 이미지당 평균 지연시간을 함께 기록한다."""
    names = names or [f"image_{i + 1}" for i in range(len(images))]
    pixel_values = preprocess(images)
    logits, elapsed_ms = infer_batched(backend, pixel_values, batch_size, on_batch)
    probs = softmax(logits)
    per_image = elapsed_ms / len(images)
    return [
        ImageResult(name=name, probs=probs[i], latency_ms=per_image) for i, name in enumerate(names)
    ]
