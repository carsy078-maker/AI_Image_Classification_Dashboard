"""공통 사이드바 — 백엔드 선택과 표시 설정.

쓸 수 없는 변형도 사유와 함께 보여준다. 환경에 따라 가용 백엔드가 달라지는 것과
그 폴백 설계가 이 프로젝트의 내용 자체라, 숨기면 보여줄 게 없어진다.
"""

from __future__ import annotations

import streamlit as st

from .. import MODEL_NAME, all_statuses, pick_default_variant
from ..config import cpu_threads
from . import theme
from .state import Settings


def render() -> Settings | None:
    """사이드바를 그리고 선택값을 돌려준다. 쓸 수 있는 백엔드가 없으면 None."""
    statuses = all_statuses()
    usable = [s for s in statuses if s.available]
    blocked = [s for s in statuses if not s.available]

    if not usable:
        st.error(
            "쓸 수 있는 추론 백엔드가 없습니다.\n\n"
            "`python scripts/export_models.py` 를 실행해 가중치를 만든 뒤 다시 열어주세요."
        )
        theme.unavailable_list([(s.label, s.reason) for s in blocked])
        return None

    with st.sidebar:
        st.html(theme.eyebrow("추론 백엔드"))

        keys = [s.key for s in usable]
        labels = {s.key: s.label for s in usable}
        default = pick_default_variant()
        index = keys.index(default) if default in keys else 0

        variant_key = st.radio(
            "추론 백엔드",
            keys,
            index=index,
            format_func=lambda k: labels[k],
            label_visibility="collapsed",
            key="vd_variant",
        )

        picked = next(s for s in usable if s.key == variant_key)
        meta = picked.source_ko
        if picked.size_mb is not None:
            meta += f" · {picked.size_mb:,.1f} MB"
        st.caption(meta)

        if blocked:
            st.html('<div style="height:0.4rem"></div>')
            st.html(theme.eyebrow("사용할 수 없음"))
            theme.unavailable_list([(s.short, s.reason) for s in blocked])

        st.divider()

        st.html(theme.eyebrow("표시 설정"))
        # 이 둘은 표시 시점에만 쓰인다. 바꿔도 추론은 다시 돌지 않는다.
        use_emoji = st.toggle("라벨에 이모지 표시", value=True, key="vd_emoji")
        top_k = st.slider("Top-K 표시 개수", min_value=1, max_value=10, value=5, key="vd_topk")

        st.divider()
        st.caption(f"모델 `{MODEL_NAME}`")
        st.caption(f"추론 스레드 {cpu_threads()}개 (PyTorch · ONNX 동일)")

    return Settings(
        variant_key=variant_key,
        variant_label=picked.label,
        use_emoji=use_emoji,
        top_k=top_k,
    )
