"""세션 상태와 추론 결과 캐시.

Streamlit 은 위젯이 하나 바뀔 때마다 스크립트를 처음부터 다시 실행한다.
결과를 들고 있지 않으면 이모지 토글 한 번에 업로드한 이미지 전부를 다시 추론하게 된다.

그래서 (백엔드, 이미지 바이트 해시) 를 키로 `ImageResult` 를 통째로 저장한다.
`ImageResult` 는 확률 1000개를 다 갖고 있으므로 Top-K 개수를 바꾸는 것은
`top(k)` 를 다시 부르는 문제일 뿐 추론과 무관하다.

재추론이 필요한 경우는 둘뿐이다: 새 이미지가 들어왔을 때, 백엔드를 바꿨을 때.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Callable
from dataclasses import dataclass

import streamlit as st

from .. import (
    ImageLoadError,
    ImageResult,
    classify,
    load_backend,
    load_image,
)

_CACHE_KEY = "vd_results"


@dataclass(frozen=True)
class Settings:
    """사이드바가 정한 값. 이 셋 중 재추론을 부르는 것은 variant_key 뿐이다."""

    variant_key: str
    variant_label: str
    use_emoji: bool
    top_k: int


def digest(blob: bytes) -> str:
    return hashlib.sha1(blob).hexdigest()


def _cache() -> dict:
    return st.session_state.setdefault(_CACHE_KEY, {})


def cached_result(variant_key: str, blob: bytes) -> ImageResult | None:
    """이미 추론해 둔 결과가 있으면 돌려준다. 없으면 None."""
    entry = _cache().get((variant_key, digest(blob)))
    return entry if isinstance(entry, ImageResult) else None


def all_cached(variant_key: str, items: list[tuple[str, bytes]]) -> bool:
    """이 목록 전부가 이미 처리된 상태인지.

    True 면 `get_or_infer` 를 불러도 추론은 한 번도 일어나지 않는다. 화면을 다시
    그려야 하는지와 추론을 돌려야 하는지를 호출한 쪽에서 구분하기 위한 것이다.
    """
    cache = _cache()
    return bool(items) and all((variant_key, digest(blob)) in cache for _, blob in items)


def get_or_infer(
    variant_key: str,
    items: list[tuple[str, bytes]],
    on_progress: Callable[[float, str], None] | None = None,
) -> tuple[list[ImageResult], list[tuple[str, str]]]:
    """캐시에 없는 것만 추론한다.

    돌려주는 값은 (성공한 결과들, 실패 목록). 실패 목록은 (파일명, 사유) 라서
    호출한 쪽이 경고로 띄우고 나머지 결과는 그대로 쓸 수 있다.

    손상된 파일 하나 때문에 전체가 죽으면 안 되므로 로드 실패는 예외로 올리지 않고
    캐시에 사유 문자열로 남긴다. 같은 파일을 다시 올려도 또 열어보지 않는다.
    """
    cache = _cache()
    keys = [(variant_key, digest(blob)) for _, blob in items]
    missing = [
        (name, blob, key) for (name, blob), key in zip(items, keys, strict=True) if key not in cache
    ]

    if missing:
        _infer_missing(variant_key, missing, on_progress)

    results: list[ImageResult] = []
    failures: list[tuple[str, str]] = []
    for (name, _), key in zip(items, keys, strict=True):
        entry = cache.get(key)
        if isinstance(entry, ImageResult):
            results.append(entry)
        else:
            failures.append((name, str(entry)))
    return results, failures


def _infer_missing(
    variant_key: str,
    missing: list[tuple[str, bytes, tuple[str, str]]],
    on_progress: Callable[[float, str], None] | None,
) -> None:
    cache = _cache()

    # 1) 먼저 전부 열어본다. 열리지 않는 파일은 사유만 남기고 빠진다.
    loaded: list[tuple[str, tuple[str, str], object]] = []
    for name, _blob, key in missing:
        try:
            loaded.append((name, key, load_image(io.BytesIO(_blob), name)))
        except ImageLoadError as exc:
            cache[key] = str(exc)

    if not loaded:
        return

    backend = load_backend(variant_key)

    # 2) 한 번에 넘긴다. 청크 분할과 진행률 보고는 코어가 처리한다.
    def report(done: int, total: int) -> None:
        if on_progress:
            on_progress(done / total, f"{done}/{total}장 분석 완료")

    outputs = classify(
        backend,
        [img for _, _, img in loaded],
        [name for name, _, _ in loaded],
        on_batch=report,
    )
    for (_, key, _), result in zip(loaded, outputs, strict=True):
        cache[key] = result


def clear_results() -> None:
    """캐시를 비운다. 테스트와 '다시 분석' 조작에서 쓴다."""
    st.session_state.pop(_CACHE_KEY, None)
