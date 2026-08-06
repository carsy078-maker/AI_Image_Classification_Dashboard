"""탭 2 — 백엔드 비교.

같은 이미지를 여러 백엔드에 통과시켜 "얼마나 빨라지는 대신 무엇을 잃는가" 를 보여준다.
화면에 나오는 배율·절감률은 전부 이 자리에서 측정한 값으로 계산한다.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from .. import (
    ImageResult,
    available_keys,
    decorate,
    variant_status,
)
from ..config import VARIANTS
from . import theme
from .state import Settings, all_cached, get_or_infer

_TOP_N = 5


def render(settings: Settings) -> None:
    usable = available_keys()
    if len(usable) < 2:
        st.info(
            "비교하려면 백엔드가 둘 이상 필요합니다. "
            "`python scripts/export_models.py` 로 ONNX 변형을 만들면 비교할 수 있습니다."
        )
        return

    item = _single_input()
    keys = _pick_backends(usable)

    if item is None:
        st.info("이미지 한 장을 올리면 선택한 백엔드에 차례로 통과시킵니다.")
        return
    if not keys:
        st.warning("비교할 백엔드를 하나 이상 선택해주세요.")
        return

    st.divider()
    run = st.button(
        "비교 실행",
        type="primary",
        key="vd_run_compare",
        help=f"{len(keys)}개 백엔드로 같은 이미지를 추론합니다",
    )

    pending = [k for k in keys if not all_cached(k, [item])]
    if not (run or not pending):
        st.caption(
            f"비교 실행을 누르면 결과가 표시됩니다. "
            f"({len(pending)}개 백엔드는 최초 로드에 시간이 걸립니다)"
        )
        return

    results = _run(keys, item, show_progress=run)
    if len(results) < 2:
        st.warning("비교할 수 있는 결과가 부족합니다.")
        return

    baseline_key = _baseline_key(list(results))
    rows = _build_rows(results, baseline_key, settings)

    _verdict(rows, baseline_key, settings)
    st.html('<div style="height:1.4rem"></div>')
    _cards(rows, settings)
    st.html('<div style="height:1.6rem"></div>')

    latency_column, dist_column = st.columns([1, 1.35], gap="large")
    with latency_column:
        theme.section("이미지당 지연시간", "짧을수록 빠르다")
        theme.chart(_latency_chart(rows), height=max(120, 46 * len(rows)))
    with dist_column:
        theme.section(
            "Top-5 확률 분포",
            f"기준선({VARIANTS[baseline_key]['short']})의 상위 5개 라벨을 축으로 맞춰 비교",
        )
        theme.chart(_distribution_chart(results, baseline_key, settings), height=260)


# --- 입력 -------------------------------------------------------------------


def _single_input() -> tuple[str, bytes] | None:
    uploaded = st.file_uploader(
        "비교할 이미지",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        key="vd_compare_upload",
    )
    if uploaded is None:
        return None
    return uploaded.name, uploaded.getvalue()


def _pick_backends(usable: list[str]) -> list[str]:
    labels = {k: VARIANTS[k]["label"] for k in usable}
    return st.multiselect(
        "비교할 백엔드",
        usable,
        default=usable,
        format_func=lambda k: labels[k],
        key="vd_compare_backends",
    )


def _run(keys: list[str], item: tuple[str, bytes], show_progress: bool) -> dict[str, ImageResult]:
    """백엔드별로 같은 이미지를 추론한다. 이미 캐시된 백엔드는 건너뛴다."""
    results: dict[str, ImageResult] = {}

    if not show_progress:
        for key in keys:
            done, _ = get_or_infer(key, [item])
            if done:
                results[key] = done[0]
        return results

    bar = st.progress(0.0, text="준비 중")
    try:
        for index, key in enumerate(keys):
            label = VARIANTS[key]["label"]
            bar.progress(index / len(keys), text=f"{label} 추론 중")
            with st.spinner(f"{label} 백엔드를 불러오는 중입니다 (최초 1회는 오래 걸립니다)"):
                done, failures = get_or_infer(key, [item])
            for name, reason in failures:
                st.warning(f"{name} — {reason}")
            if done:
                results[key] = done[0]
    finally:
        bar.empty()
    return results


# --- 계산 -------------------------------------------------------------------


def _baseline_key(keys: list[str]) -> str:
    """기준선은 가장 무거운 FP32. 없으면 VARIANTS 정의 순서상 첫 번째."""
    ordered = [k for k in VARIANTS if k in keys]
    for key in ordered:
        if VARIANTS[key]["precision"] == "fp32":
            return key
    return ordered[0]


def _build_rows(
    results: dict[str, ImageResult], baseline_key: str, settings: Settings
) -> list[dict]:
    """표·차트·요약문이 공유하는 행 데이터. 상대 지표는 여기서 한 번만 계산한다."""
    baseline = results[baseline_key]
    baseline_top1 = baseline.best
    baseline_size = variant_status(baseline_key).size_mb

    rows = []
    for key in [k for k in VARIANTS if k in results]:
        result = results[key]
        best = result.best
        size = variant_status(key).size_mb
        rows.append(
            {
                "key": key,
                "label": VARIANTS[key]["label"],
                "short": VARIANTS[key]["short"],
                "is_baseline": key == baseline_key,
                "top1": best.label,
                "display_top1": decorate(best.label, settings.use_emoji),
                "score": best.score,
                "latency_ms": result.latency_ms,
                "speedup": baseline.latency_ms / result.latency_ms,
                "agrees": best.index == baseline_top1.index,
                "size_mb": size,
                "size_cut": ((1 - size / baseline_size) * 100 if size and baseline_size else None),
            }
        )
    return rows


# --- 표시 -------------------------------------------------------------------


def _verdict(rows: list[dict], baseline_key: str, settings: Settings) -> None:
    """가장 빠른 비기준선 백엔드를 기준선과 대비해 한 문장으로 요약한다."""
    baseline = next(r for r in rows if r["is_baseline"])
    others = [r for r in rows if not r["is_baseline"]]
    if not others:
        return

    best = max(others, key=lambda r: r["speedup"])
    parts = [
        f"<strong>{theme.esc(best['label'])}</strong>은 "
        f"{theme.esc(baseline['short'])} 대비 "
        f"<strong>{best['speedup']:.2f}배</strong> "
        f"{'빠르고' if best['speedup'] >= 1 else '느리고'}"
    ]
    if best["size_cut"] is not None:
        direction = "작으면서" if best["size_cut"] >= 0 else "크면서"
        parts.append(f"용량은 <strong>{abs(best['size_cut']):.1f}%</strong> {direction}")
    parts.append(
        "같은 답을 냈습니다."
        if best["agrees"]
        else f"답이 갈렸습니다 (<strong>{theme.esc(best['top1'])}</strong>)."
    )
    theme.verdict(" ".join(parts))

    disagreed = [r for r in others if not r["agrees"]]
    if disagreed:
        names = ", ".join(r["short"] for r in disagreed)
        st.caption(f"기준선과 다른 Top-1 을 낸 백엔드: {names}")


def _cards(rows: list[dict], settings: Settings) -> None:
    columns = st.columns(len(rows), gap="small")
    for column, row in zip(columns, rows, strict=True):
        with column, st.container(border=True):
            st.html(
                theme.eyebrow(row["short"])
                + (
                    theme.pill("기준선", "info")
                    if row["is_baseline"]
                    else theme.pill(f"{row['speedup']:.2f}× 속도", "neutral")
                )
            )
            st.html('<div style="height:0.55rem"></div>')
            st.markdown(f"**{row['display_top1']}**")
            st.html(
                theme.pill(f"{row['score']:.1%}", "positive" if row["score"] >= 0.5 else "caution")
                + "&nbsp;"
                + (
                    theme.pill("✅ 일치", "positive")
                    if row["agrees"]
                    else theme.pill("⚠️ 불일치", "caution")
                )
            )
            st.html('<div style="height:0.35rem"></div>')
            size = f"{row['size_mb']:,.1f} MB" if row["size_mb"] else "용량 정보 없음"
            st.caption(f"{row['latency_ms']:.1f} ms · {size}")


def _latency_chart(rows: list[dict]) -> alt.Chart:
    frame = pd.DataFrame(
        [
            {
                "backend": r["short"],
                "latency": r["latency_ms"],
                "color": theme.variant_color(r["key"]),
                "text": f"{r['latency_ms']:.1f} ms",
                "order": index,
            }
            for index, r in enumerate(rows)
        ]
    )
    base = alt.Chart(frame).encode(
        y=alt.Y("backend:N", sort=alt.SortField("order"), title=None),
        x=alt.X("latency:Q", title="ms", axis=alt.Axis(grid=True)),
    )
    bars = base.mark_bar(height=alt.RelativeBandSize(0.5)).encode(
        color=alt.Color("color:N", scale=None)
    )
    labels = base.mark_text(align="left", dx=6, color=theme.MUTED, fontSize=11).encode(
        text="text:N"
    )
    return (bars + labels).properties(padding={"left": 0, "right": 52, "top": 0, "bottom": 0})


def _distribution_chart(
    results: dict[str, ImageResult], baseline_key: str, settings: Settings
) -> alt.Chart:
    """기준선의 Top-5 라벨을 축으로 고정하고 각 백엔드의 같은 라벨 확률을 겹쳐 놓는다.

    백엔드마다 자기 Top-5 를 따로 그리면 축이 어긋나 비교가 되지 않는다.
    """
    axis = results[baseline_key].top(_TOP_N)

    frame = pd.DataFrame(
        [
            {
                "label": decorate(prediction.label, settings.use_emoji),
                "backend": VARIANTS[key]["short"],
                "score": float(result.probs[prediction.index]),
                "order": rank,
            }
            for rank, prediction in enumerate(axis)
            for key, result in results.items()
        ]
    )
    order = [VARIANTS[k]["short"] for k in VARIANTS if k in results]
    colors = [theme.variant_color(k) for k in VARIANTS if k in results]

    return (
        alt.Chart(frame)
        .mark_bar(height=alt.RelativeBandSize(0.78))
        .encode(
            y=alt.Y(
                "label:N", sort=alt.SortField("order"), title=None, axis=alt.Axis(labelLimit=200)
            ),
            x=alt.X(
                "score:Q",
                title=None,
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", values=[0, 0.5, 1], grid=True),
            ),
            yOffset=alt.YOffset("backend:N", sort=order),
            color=alt.Color(
                "backend:N",
                sort=order,
                scale=alt.Scale(domain=order, range=colors),
                legend=alt.Legend(title=None),
            ),
            tooltip=[
                alt.Tooltip("backend:N", title="백엔드"),
                alt.Tooltip("label:N", title="라벨"),
                alt.Tooltip("score:Q", title="확률", format=".2%"),
            ],
        )
    )
