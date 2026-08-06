"""ViT 이미지 분류 대시보드의 코어 라이브러리.

UI(Streamlit) 는 이 패키지 위에 얹히기만 하고, 추론·측정 로직은 전부 여기 있다.
Streamlit 없이도 import 되어야 한다 (CLI 스크립트와 테스트가 그렇게 쓴다).
"""

from .backends import (
    Backend,
    VariantStatus,
    all_statuses,
    available_keys,
    load_backend,
    pick_default_variant,
    variant_status,
)
from .config import DEFAULT_VARIANT, MAX_BATCH_IMAGES, MODEL_NAME, VARIANTS
from .inference import (
    ImageLoadError,
    ImageResult,
    Prediction,
    classify,
    load_image,
    preprocess,
    softmax,
    thumbnail,
)
from .labels import decorate, emoji_for, id2label, label_of

__all__ = [
    "Backend",
    "DEFAULT_VARIANT",
    "ImageLoadError",
    "ImageResult",
    "MAX_BATCH_IMAGES",
    "MODEL_NAME",
    "Prediction",
    "VARIANTS",
    "VariantStatus",
    "all_statuses",
    "available_keys",
    "classify",
    "decorate",
    "emoji_for",
    "id2label",
    "label_of",
    "load_backend",
    "load_image",
    "pick_default_variant",
    "preprocess",
    "softmax",
    "thumbnail",
    "variant_status",
]
