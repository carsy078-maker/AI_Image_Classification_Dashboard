"""탭 1 — 이미지 분류.

업로드한 이미지를 전부 미리 보여주고, 버튼을 눌렀을 때만 추론한다.
추론 이후의 표시 설정 변경(이모지, Top-K)은 캐시에서 다시 그리기만 한다.
"""

from __future__ import annotations

import io
import json

import altair as alt
import pandas as pd
import streamlit as st

from .. import (
    MAX_BATCH_IMAGES,
    ImageLoadError,
    ImageResult,
    decorate,
    load_image,
    thumbnail,
)
from . import theme
from .state import Settings, all_cached, get_or_infer

_PREVIEW_COLS = 4


@st.cache_data(show_spinner=False, max_entries=64)
def _preview_png(blob: bytes) -> bytes | None:
    """표시용 축소본을 PNG 바이트로. 원본 해상도를 브라우저로 보내지 않는다.

    바이트를 키로 캐싱해서, 슬라이더를 움직일 때마다 32장을 다시 열고 줄이는 일이
    없게 한다.
    """
    try:
        image = thumbnail(load_image(io.BytesIO(blob)))
    except ImageLoadError:
        return None
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render(settings: Settings) -> None:
    items = _collect_input()

    if not items:
        st.info("이미지를 올리면 여기에 미리보기가 나타납니다.")
        return

    _preview_grid(items)
    st.divider()

    run = st.button(
        "분석 실행",
        type="primary",
        key="vd_run_classify",
        help=f"{settings.variant_label} 백엔드로 {len(items)}장을 추론합니다",
    )

    # 버튼을 누른 적이 있거나(=이번 클릭), 이미 전부 처리된 상태일 때만 결과를 그린다.
    # 후자에서는 get_or_infer 가 캐시만 읽으므로 추론이 다시 돌지 않는다.
    if not (run or all_cached(settings.variant_key, items)):
        st.caption("분석 실행을 누르면 결과가 표시됩니다.")
        return

    results, failures = _infer(settings, items, show_progress=run)

    for name, reason in failures:
        st.warning(f"{name} — {reason}")

    if not results:
        return

    st.divider()
    _summary(settings, results)
    st.html('<div style="height:1.2rem"></div>')

    for result, (_, blob) in zip(results, _successful(items, failures), strict=True):
        _result_card(settings, result, blob)

    st.divider()
    _export(settings, results)


# --- 입력 -------------------------------------------------------------------


def _collect_input() -> list[tuple[str, bytes]]:
    """(파일명, 원본 바이트) 목록. 바이트를 그대로 들고 다녀야 캐시 키를 만들 수 있다."""
    mode = st.segmented_control(
        "입력 방식",
        ["파일 업로드", "카메라 촬영"],
        default="파일 업로드",
        key="vd_input_mode",
        label_visibility="collapsed",
    )

    if mode == "카메라 촬영":
        shot = st.camera_input("사진 촬영", key="vd_camera", label_visibility="collapsed")
        return [("camera.jpg", shot.getvalue())] if shot else []

    uploaded = st.file_uploader(
        "이미지 파일",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="vd_uploader",
        label_visibility="collapsed",
    )
    if not uploaded:
        return []

    if len(uploaded) > MAX_BATCH_IMAGES:
        st.warning(
            f"한 번에 {MAX_BATCH_IMAGES}장까지 처리합니다. "
            f"올린 {len(uploaded)}장 중 뒤쪽 {len(uploaded) - MAX_BATCH_IMAGES}장은 제외했습니다."
        )
        uploaded = uploaded[:MAX_BATCH_IMAGES]

    return [(f.name, f.getvalue()) for f in uploaded]


def _preview_grid(items: list[tuple[str, bytes]]) -> None:
    st.html(theme.eyebrow(f"입력 {len(items)}장"))
    for start in range(0, len(items), _PREVIEW_COLS):
        row = items[start : start + _PREVIEW_COLS]
        columns = st.columns(_PREVIEW_COLS, gap="small")
        # 마지막 행은 열 수보다 이미지가 적을 수 있으므로 짧은 쪽에 맞춘다.
        for column, (name, blob) in zip(columns, row, strict=False):
            with column:
                png = _preview_png(blob)
                if png is None:
                    st.caption(f"{name} — 열 수 없음")
                else:
                    st.image(png, caption=name, width="stretch")


# --- 추론 -------------------------------------------------------------------


def _infer(
    settings: Settings, items: list[tuple[str, bytes]], show_progress: bool
) -> tuple[list[ImageResult], list[tuple[str, str]]]:
    if not show_progress:
        return get_or_infer(settings.variant_key, items)

    bar = st.progress(0.0, text="백엔드 준비 중")
    try:
        spinner_text = (
            f"{settings.variant_label} 백엔드를 불러오는 중입니다 (최초 1회는 오래 걸립니다)"
        )
        with st.spinner(spinner_text):
            results, failures = get_or_infer(
                settings.variant_key,
                items,
                on_progress=lambda ratio, text: bar.progress(ratio, text=text),
            )
    finally:
        bar.empty()
    return results, failures


def _successful(
    items: list[tuple[str, bytes]], failures: list[tuple[str, str]]
) -> list[tuple[str, bytes]]:
    failed = {name for name, _ in failures}
    return [item for item in items if item[0] not in failed]


# --- 결과 -------------------------------------------------------------------


def _summary(settings: Settings, results: list[ImageResult]) -> None:
    scores = [r.best.score for r in results]
    latencies = [r.latency_ms for r in results]
    color = theme.variant_color(settings.variant_key)
    theme.stat_strip(
        [
            {"label": "분석한 이미지", "value": f"{len(results)}", "unit": "장"},
            {
                "label": "평균 신뢰도",
                "value": f"{sum(scores) / len(scores):.1%}",
                "foot": f"최저 {min(scores):.1%} · 최고 {max(scores):.1%}",
            },
            {
                "label": "이미지당 지연시간",
                "value": f"{sum(latencies) / len(latencies):.1f}",
                "unit": "ms",
                "color": color,
                "foot": theme.esc(settings.variant_label),
            },
        ]
    )


def _result_card(settings: Settings, result: ImageResult, blob: bytes) -> None:
    predictions = result.top(settings.top_k)
    best = predictions[0]

    with st.container(border=True):
        left, right = st.columns([1, 2.4], gap="medium", vertical_alignment="top")

        with left:
            png = _preview_png(blob)
            if png is not None:
                st.image(png, width="stretch")

        with right:
            st.html(theme.eyebrow(result.name))
            st.markdown(f"### {decorate(best.label, settings.use_emoji)}")
            st.html(
                f"{theme.pill(f'{best.score:.1%}', 'positive' if best.score >= 0.5 else 'caution')}"
                f"&nbsp;{theme.pill(f'{result.latency_ms:.1f} ms', 'neutral')}"
            )
            st.html('<div style="height:0.85rem"></div>')
            theme.chart(
                _topk_chart(settings, predictions),
                height=max(90, 34 * len(predictions)),
            )


def _topk_chart(settings: Settings, predictions: list) -> alt.Chart:
    """Top-K 가로 막대. 값 텍스트를 막대 끝에 붙여 축을 읽지 않아도 되게 한다."""
    frame = pd.DataFrame(
        {
            "label": [decorate(p.label, settings.use_emoji) for p in predictions],
            "score": [p.score for p in predictions],
            "order": list(range(len(predictions))),
        }
    )
    base = alt.Chart(frame).encode(
        y=alt.Y("label:N", sort=alt.SortField("order"), title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X(
            "score:Q",
            title=None,
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(format="%", values=[0, 0.5, 1], grid=True),
        ),
    )
    bars = base.mark_bar(
        color=theme.variant_color(settings.variant_key), height=alt.RelativeBandSize(0.52)
    )
    labels = base.mark_text(align="left", dx=6, color=theme.MUTED, fontSize=11).encode(
        text=alt.Text("score:Q", format=".1%")
    )
    return (bars + labels).properties(padding={"left": 0, "right": 46, "top": 0, "bottom": 0})


# --- 내보내기 ---------------------------------------------------------------


def _export(settings: Settings, results: list[ImageResult]) -> None:
    rows = [
        {
            "image": result.name,
            "rank": rank,
            "label": prediction.label,
            "score": round(prediction.score, 6),
            "variant": settings.variant_key,
            "latency_ms": round(result.latency_ms, 3),
        }
        for result in results
        for rank, prediction in enumerate(result.top(settings.top_k), start=1)
    ]

    st.html(theme.eyebrow("결과 내보내기"))
    csv_column, json_column = st.columns(2, gap="small")
    with csv_column:
        st.download_button(
            "CSV 내려받기",
            # Excel 이 UTF-8 을 알아보게 BOM 을 붙인다.
            data=pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig"),
            file_name=f"classification_{settings.variant_key}.csv",
            mime="text/csv",
            width="stretch",
        )
    with json_column:
        st.download_button(
            "JSON 내려받기",
            data=json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"classification_{settings.variant_key}.json",
            mime="application/json",
            width="stretch",
        )
