"""추론 백엔드 추상화.

같은 ViT 모델을 런타임(PyTorch / ONNX Runtime)과 정밀도(FP32 / INT8) 조합으로
바꿔 끼울 수 있게 감싼다. 호출자는 `load_backend(key).infer(x)` 만 알면 된다.

가중치 확보 경로는 환경에 따라 세 갈래다.

    1. 로컬 models/          scripts/export_models.py 로 내보낸 파일
    2. Hugging Face Hub      배포 환경에서 ONNX INT8(84MB) 만 내려받는다
    3. 원본에서 즉석 구성     PyTorch 변형은 파일 없이도 from_pretrained 로 만든다

330MB 짜리 FP32 파일을 배포 이미지에 넣을 수 없으므로, 쓸 수 있는 백엔드가
환경마다 달라진다. 그 판정을 `variant_status()` 한곳에 모아두고 UI 는 결과만 읽는다.
"""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import (
    HUB_ONNX_REPO,
    MODEL_NAME,
    VARIANTS,
    cpu_threads,
)


@dataclass(frozen=True)
class VariantStatus:
    """한 변형을 현재 환경에서 쓸 수 있는지에 대한 판정 결과."""

    key: str
    label: str
    short: str
    runtime: str
    precision: str
    available: bool
    source: str  # "local" | "hub" | "pretrained" | "-"
    reason: str
    size_mb: float | None

    @property
    def source_ko(self) -> str:
        return {
            "local": "로컬 파일",
            "hub": "Hugging Face Hub",
            "pretrained": "원본에서 즉석 구성",
        }.get(self.source, "-")


def _size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 * 1024), 2)


def _probe_quantization(engine: str) -> bool:
    """이 엔진으로 실제 양자화가 되는지 작은 레이어로 시험해 본다."""
    import torch

    try:
        torch.backends.quantized.engine = engine
        tiny = torch.nn.Sequential(torch.nn.Linear(4, 2))
        quantized = torch.ao.quantization.quantize_dynamic(
            tiny, {torch.nn.Linear}, dtype=torch.qint8
        )
        quantized(torch.zeros(1, 4))
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def setup_quantization_engine() -> str:
    """동적 양자화에 쓸 백엔드 엔진을 고른다.

    `torch.backends.quantized.supported_engines` 를 믿으면 안 된다. Linux
    aarch64 에서도 ['qnnpack', 'onednn', 'x86', 'fbgemm'] 을 전부 보고하지만
    x86 과 fbgemm 은 호출하는 순간 `RuntimeError: unknown architecure` 로
    죽는다. 게다가 그 환경의 기본 엔진이 하필 x86 이라, 설정된 값을 그대로
    쓰면 실패한다 (macOS 는 반대로 기본값이 'none' 이라 이 함정이 안 보인다).

    그래서 목록을 믿지 않고 후보를 실제로 양자화해 보며 고른다. 시험용 레이어는
    Linear(4, 2) 라 비용이 사실상 없고, 결과는 프로세스 수명 동안 캐시한다.
    """
    import platform

    import torch

    supported = list(torch.backends.quantized.supported_engines)
    current = torch.backends.quantized.engine

    # 아키텍처에 맞는 것을 앞에 둔다. 현재 설정값은 신뢰하지 않고 검증 대상에만 넣는다.
    if platform.machine().lower() in ("aarch64", "arm64"):
        preferred = ("qnnpack", "onednn")
    else:
        preferred = ("fbgemm", "x86", "onednn", "qnnpack")

    seen: set[str] = set()
    candidates = [e for e in (*preferred, current, *supported) if e and e != "none"]

    for engine in candidates:
        if engine in seen or engine not in supported:
            continue
        seen.add(engine)
        if _probe_quantization(engine):
            return engine

    raise RuntimeError(f"이 환경에서 동작하는 양자화 엔진이 없다. 보고된 목록: {supported}")


def torch_installed() -> bool:
    """torch 설치 여부. import 하지 않고 확인한다 (import 만으로 수 초가 걸린다).

    배포 이미지에서는 torch(수백 MB)를 빼고 ONNX Runtime 만 싣는다. 그 환경에서
    PyTorch 변형은 애초에 선택지가 될 수 없다.
    """
    return importlib.util.find_spec("torch") is not None


def variant_status(key: str) -> VariantStatus:
    """가중치를 어디서 얻을 수 있는지 판정한다. 네트워크 요청은 하지 않는다."""
    meta = VARIANTS[key]
    path: Path = meta["path"]
    common = {
        "key": key,
        "label": meta["label"],
        "short": meta["short"],
        "runtime": meta["runtime"],
        "precision": meta["precision"],
    }

    if meta["runtime"] == "pytorch" and not torch_installed():
        return VariantStatus(
            **common,
            available=False,
            source="-",
            reason="이 환경에는 torch 가 없다 (ONNX 전용 배포)",
            size_mb=_size_mb(path) if path.exists() else None,
        )

    if path.exists():
        return VariantStatus(
            **common,
            available=True,
            source="local",
            reason=f"{path.name}",
            size_mb=_size_mb(path),
        )

    if meta["hub_filename"] and HUB_ONNX_REPO:
        return VariantStatus(
            **common,
            available=True,
            source="hub",
            reason=f"{HUB_ONNX_REPO} 에서 내려받는다",
            size_mb=None,
        )

    if not meta["needs_file"]:
        return VariantStatus(
            **common,
            available=True,
            source="pretrained",
            reason=f"{MODEL_NAME} 원본에서 구성한다",
            size_mb=None,
        )

    return VariantStatus(
        **common,
        available=False,
        source="-",
        reason="scripts/export_models.py 를 실행해 가중치를 만들어야 한다",
        size_mb=None,
    )


def all_statuses() -> list[VariantStatus]:
    return [variant_status(k) for k in VARIANTS]


def available_keys() -> list[str]:
    return [s.key for s in all_statuses() if s.available]


def resolve_weight_path(key: str) -> Path | None:
    """가중치 파일의 실제 경로. 필요하면 Hub 에서 내려받는다.

    PyTorch 변형은 파일 없이도 동작하므로 None 을 돌려줄 수 있다.
    """
    meta = VARIANTS[key]
    path: Path = meta["path"]
    if path.exists():
        return path

    if meta["hub_filename"] and HUB_ONNX_REPO:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(repo_id=HUB_ONNX_REPO, filename=meta["hub_filename"])
        return Path(downloaded)

    if meta["needs_file"]:
        raise FileNotFoundError(
            f"{meta['label']} 가중치를 찾을 수 없다: {path}\n"
            "scripts/export_models.py 를 먼저 실행할 것."
        )
    return None


class Backend(ABC):
    """(N, 3, 224, 224) float32 를 받아 (N, 1000) logits 를 돌려주는 추론기."""

    def __init__(self, key: str) -> None:
        self.key = key
        self.meta = VARIANTS[key]
        self.label = self.meta["label"]

    @abstractmethod
    def infer(self, pixel_values: np.ndarray) -> np.ndarray: ...


class TorchBackend(Backend):
    def __init__(self, key: str) -> None:
        super().__init__(key)
        import torch
        from transformers import ViTForImageClassification

        torch.set_num_threads(cpu_threads())
        self._torch = torch

        model = ViTForImageClassification.from_pretrained(MODEL_NAME)
        model.eval()

        if self.meta["precision"] == "int8":
            setup_quantization_engine()
            # 양자화된 state_dict 는 FP32 구조에 그대로 못 넣는다. 먼저 같은
            # 양자화를 적용해 모듈 구조를 맞춘 뒤 로드해야 한다.
            model = torch.ao.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )

        path = resolve_weight_path(key)
        if path is not None:
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()

        self._model = model

    def infer(self, pixel_values: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            out = self._model(pixel_values=torch.from_numpy(pixel_values))
        return out.logits.numpy()


class OnnxBackend(Backend):
    def __init__(self, key: str) -> None:
        super().__init__(key)
        import onnxruntime as ort

        path = resolve_weight_path(key)
        opts = ort.SessionOptions()
        # PyTorch 쪽과 스레드 수를 맞춰야 런타임 비교가 성립한다.
        opts.intra_op_num_threads = cpu_threads()
        opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    def infer(self, pixel_values: np.ndarray) -> np.ndarray:
        return self._session.run(None, {self._input_name: pixel_values})[0]


@lru_cache(maxsize=4)
def load_backend(key: str) -> Backend:
    """백엔드를 로드한다. 같은 key 는 프로세스 수명 동안 재사용한다.

    Streamlit 은 위젯이 바뀔 때마다 스크립트를 처음부터 다시 실행하므로,
    캐싱하지 않으면 슬라이더 한 번에 수백 MB 를 다시 읽게 된다.
    """
    if key not in VARIANTS:
        raise KeyError(f"알 수 없는 변형: {key}")
    runtime = VARIANTS[key]["runtime"]
    return TorchBackend(key) if runtime == "pytorch" else OnnxBackend(key)


def pick_default_variant() -> str | None:
    """현재 환경에서 기본으로 쓸 변형. 가벼운 쪽을 우선한다."""
    from .config import VARIANT_PREFERENCE

    usable = set(available_keys())
    for key in VARIANT_PREFERENCE:
        if key in usable:
            return key
    return None
