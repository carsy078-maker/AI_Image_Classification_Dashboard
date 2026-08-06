"""전역 설정 - 경로, 모델 변형 정의, 실행 환경 프로파일."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
RESULTS_DIR = ROOT / "results"

MODEL_NAME = "google/vit-base-patch16-224"
IMAGE_SIZE = 224
NUM_CLASSES = 1000
OPSET_VERSION = 14

# 배포 환경에서 ONNX INT8 가중치를 내려받을 Hugging Face 저장소.
# 비워두면 Hub 폴백이 비활성화되고 로컬 models/ 만 탐색한다.
HUB_ONNX_REPO = os.getenv("VD_HUB_ONNX_REPO", "")

# 업로드 이미지 1장당 허용 픽셀 수 상한. 초과하면 비율을 유지한 채 축소한다.
# 원본 그대로 브라우저로 보내면 썸네일 하나에 수 MB 가 오간다.
MAX_PREVIEW_PIXELS = 1_200_000

# 한 번에 처리할 이미지 수 상한. 무제한으로 열어두면 브라우저와 서버 양쪽이 죽는다.
MAX_BATCH_IMAGES = 32

# 추론 배치 크기. ONNX 그래프의 배치 축이 동적으로 열려 있어 조정 가능하다.
INFER_BATCH_SIZE = 8


# 벤치마크·추론 대상 4종. key 는 결과 파일과 UI 전반에서 식별자로 쓴다.
#
# needs_file      True  = 가중치 파일이 있어야만 쓸 수 있다
#                 False = 파일이 없으면 허브 원본에서 즉석 구성한다
# hub_filename    HUB_ONNX_REPO 에서 내려받을 파일명 (없으면 Hub 폴백 대상 아님)
VARIANTS: dict[str, dict] = {
    "pytorch_fp32": {
        "label": "PyTorch FP32",
        "short": "PT FP32",
        "path": MODELS_DIR / "vit_fp32.pth",
        "runtime": "pytorch",
        "precision": "fp32",
        "needs_file": False,
        "hub_filename": None,
        "note": "기준선. transformers 원본 가중치를 그대로 사용한다.",
    },
    "pytorch_int8": {
        "label": "PyTorch INT8 (dynamic)",
        "short": "PT INT8",
        "path": MODELS_DIR / "vit_int8_dynamic.pth",
        "runtime": "pytorch",
        "precision": "int8",
        "needs_file": False,
        "hub_filename": None,
        "note": "Linear 레이어만 동적 양자화. 파일이 없으면 로드 시점에 양자화한다.",
    },
    "onnx_fp32": {
        "label": "ONNX FP32",
        "short": "ONNX FP32",
        "path": MODELS_DIR / "vit_fp32.onnx",
        "runtime": "onnx",
        "precision": "fp32",
        "needs_file": True,
        "hub_filename": None,
        "note": "330MB 라 배포 이미지에 포함하지 않는다. 로컬 export 전용.",
    },
    "onnx_int8": {
        "label": "ONNX INT8 (dynamic)",
        "short": "ONNX INT8",
        "path": MODELS_DIR / "vit_int8_dynamic.onnx",
        "runtime": "onnx",
        "precision": "int8",
        "needs_file": True,
        "hub_filename": "vit_int8_dynamic.onnx",
        "note": "84MB. 배포 환경의 기본 백엔드.",
    },
}

DEFAULT_VARIANT = "onnx_int8"

# 배포 환경에서 우선 시도할 순서. 앞쪽이 가벼운 백엔드다.
VARIANT_PREFERENCE = ("onnx_int8", "onnx_fp32", "pytorch_int8", "pytorch_fp32")


def ensure_dirs() -> None:
    for d in (MODELS_DIR, DATA_DIR, SAMPLES_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def cpu_threads() -> int:
    """추론에 쓸 스레드 수.

    PyTorch 와 ONNX Runtime 에 같은 값을 강제해야 런타임 비교가 공정해진다.
    기본값을 그대로 두면 각자 다른 수의 코어를 잡아 '런타임 비교' 가 아니라
    '스레드 수 비교' 가 되어 버린다.
    """
    env = os.getenv("VD_CPU_THREADS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, min(4, os.cpu_count() or 1))
