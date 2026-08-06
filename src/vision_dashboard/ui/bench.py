"""탭 3 — 벤치마크 결과.

`scripts/run_benchmark.py` 가 남긴 측정 결과를 읽어 보여주기만 한다.
이 화면에서 재측정은 하지 않는다 (200장 측정은 수 분이 걸려 세션에 부담이 크다).
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from ..benchmarking import RESULTS_PATH, comparison_rows, load_results
from ..config import VARIANTS
from . import theme


def render() -> None:
    payload = load_results()
    if payload is None:
        _empty_state()
        return

    rows = comparison_rows(payload)
    _headline(payload, rows)
    st.html('<div style="height:1.4rem"></div>')
    _environment(payload)
    st.html('<div style="height:1.2rem"></div>')
    _table(rows)
    st.html('<div style="height:1.6rem"></div>')
    _charts(payload, rows)


def _empty_state() -> None:
    """클론 직후에는 결과 파일이 없다. 이 분기가 없으면 첫 실행에서 앱이 죽는다."""
    st.info(
        "아직 측정 결과가 없습니다.\n\n아래 명령으로 벤치마크를 돌리면 이 탭에 표가 채워집니다."
    )
    st.code("python scripts/run_benchmark.py --n 200", language="bash")
    st.caption(f"결과는 `{RESULTS_PATH.relative_to(RESULTS_PATH.parents[1])}` 에 저장됩니다.")


def _headline(payload: dict, rows: list[dict]) -> None:
    """가장 빠른 변형과 가장 정확한 변형을 먼저 보여준다."""
    fastest = min(rows, key=lambda r: r["지연 mean(ms)"])
    most_accurate = max(rows, key=lambda r: r["Top-1(%)"])
    sized = [r for r in rows if r["용량(MB)"]]
    smallest = min(sized, key=lambda r: r["용량(MB)"]) if sized else None

    cells = [
        {
            "label": "가장 빠른 변형",
            "value": f"{fastest['지연 mean(ms)']:,.1f}",
            "unit": "ms",
            "color": theme.variant_color(fastest["key"]),
            "foot": theme.esc(f"{fastest['변형']} · {fastest['속도']:.2f}× 기준선 대비"),
        },
        {
            "label": "가장 정확한 변형",
            "value": f"{most_accurate['Top-1(%)']:.1f}",
            "unit": "%",
            "color": theme.variant_color(most_accurate["key"]),
            "foot": theme.esc(f"{most_accurate['변형']} · Top-5 {most_accurate['Top-5(%)']:.1f}%"),
        },
    ]
    if smallest is not None:
        cut = smallest["용량 절감(%)"]
        cells.append(
            {
                "label": "가장 작은 변형",
                "value": f"{smallest['용량(MB)']:,.0f}",
                "unit": "MB",
                "color": theme.variant_color(smallest["key"]),
                "foot": theme.esc(
                    f"{smallest['변형']}"
                    + (f" · 기준선 대비 {cut:.1f}% 절감" if cut is not None else "")
                ),
            }
        )
    theme.stat_strip(cells)


def _environment(payload: dict) -> None:
    """다른 CPU 에서 잰 수치는 비교 의미가 없다. 표에 항상 붙어 있어야 한다."""
    env = payload["environment"]
    st.caption(
        f"평가 이미지 {payload['num_eval_images']}장 · 배치 {payload['batch_size']} · "
        f"모델 `{payload['model']}`"
    )
    st.caption(
        f"측정 환경 — {env['platform']} · {env['processor']} · "
        f"Python {env['python']} · CPU {env['cpu_threads']} threads (PyTorch·ONNX 동일)"
    )


def _table(rows: list[dict]) -> None:
    frame = pd.DataFrame(rows).drop(columns=["key"])
    st.dataframe(
        frame,
        hide_index=True,
        column_config={
            "변형": st.column_config.TextColumn(width="medium"),
            "용량(MB)": st.column_config.NumberColumn(format="%.1f MB"),
            "용량 절감(%)": st.column_config.NumberColumn(format="%.1f%%"),
            "지연 mean(ms)": st.column_config.NumberColumn(format="%.1f ms"),
            "지연 p95(ms)": st.column_config.NumberColumn(format="%.1f ms"),
            "속도": st.column_config.NumberColumn(format="%.2f×"),
            "Top-1(%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Top-5(%)": st.column_config.NumberColumn(format="%.2f%%"),
            "FP32 일치(%)": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )


def _charts(payload: dict, rows: list[dict]) -> None:
    size_column, accuracy_column = st.columns(2, gap="large")

    with size_column:
        theme.section("모델 용량", "로컬 가중치 파일이 있는 변형만")
        sized = [r for r in rows if r["용량(MB)"]]
        if sized:
            theme.chart(
                _bar(sized, "용량(MB)", "MB", "{:,.0f} MB"),
                height=max(120, 46 * len(sized)),
            )
        else:
            st.caption("로컬 가중치 파일이 없어 용량을 비교할 수 없습니다.")

    with accuracy_column:
        theme.section("Top-1 정확도", f"평가 이미지 {payload['num_eval_images']}장 기준")
        theme.chart(
            _accuracy_chart(rows),
            height=max(120, 46 * len(rows)),
        )

    theme.section("지연시간", "단일 이미지 추론. mean 과 p95 를 함께 본다")
    theme.chart(_latency_chart(rows), height=max(150, 62 * len(rows)))


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "backend": VARIANTS[r["key"]]["short"],
                "color": theme.variant_color(r["key"]),
                "order": index,
                **{k: v for k, v in r.items() if k not in ("key", "변형")},
            }
            for index, r in enumerate(rows)
        ]
    )


def _bar(rows: list[dict], field: str, axis_title: str, text_format: str) -> alt.Chart:
    frame = _frame(rows)
    frame["text"] = frame[field].map(lambda v: text_format.format(v))
    base = alt.Chart(frame).encode(
        y=alt.Y("backend:N", sort=alt.SortField("order"), title=None),
        x=alt.X(f"{field}:Q", title=axis_title, axis=alt.Axis(grid=True)),
    )
    bars = base.mark_bar(height=alt.RelativeBandSize(0.5)).encode(
        color=alt.Color("color:N", scale=None)
    )
    labels = base.mark_text(align="left", dx=6, color=theme.MUTED, fontSize=11).encode(
        text="text:N"
    )
    return (bars + labels).properties(padding={"left": 0, "right": 64, "top": 0, "bottom": 0})


def _accuracy_chart(rows: list[dict]) -> alt.Chart:
    """정확도 차이는 소수점 단위라 0 부터 그리면 막대가 전부 같아 보인다.

    변형 간 차이가 드러나도록 x 축을 실제 값 주변으로 좁힌다.
    """
    frame = _frame(rows)
    frame["text"] = frame["Top-1(%)"].map("{:.2f}%".format)
    low = max(0.0, frame["Top-1(%)"].min() - 2)
    high = min(100.0, frame["Top-1(%)"].max() + 1)

    base = alt.Chart(frame).encode(
        y=alt.Y("backend:N", sort=alt.SortField("order"), title=None),
        x=alt.X(
            "Top-1(%):Q",
            title="Top-1 %",
            scale=alt.Scale(domain=[low, high], clamp=True),
            axis=alt.Axis(grid=True),
        ),
    )
    bars = base.mark_bar(height=alt.RelativeBandSize(0.5)).encode(
        color=alt.Color("color:N", scale=None)
    )
    labels = base.mark_text(align="left", dx=6, color=theme.MUTED, fontSize=11).encode(
        text="text:N"
    )
    return (bars + labels).properties(padding={"left": 0, "right": 60, "top": 0, "bottom": 0})


def _latency_chart(rows: list[dict]) -> alt.Chart:
    frame = _frame(rows).melt(
        id_vars=["backend", "order"],
        value_vars=["지연 mean(ms)", "지연 p95(ms)"],
        var_name="지표",
        value_name="ms",
    )
    frame["지표"] = frame["지표"].map({"지연 mean(ms)": "mean", "지연 p95(ms)": "p95"})

    return (
        alt.Chart(frame)
        .mark_bar(height=alt.RelativeBandSize(0.8))
        .encode(
            y=alt.Y("backend:N", sort=alt.SortField("order"), title=None),
            x=alt.X("ms:Q", title="ms", axis=alt.Axis(grid=True)),
            yOffset=alt.YOffset("지표:N", sort=["mean", "p95"]),
            color=alt.Color(
                "지표:N",
                sort=["mean", "p95"],
                scale=alt.Scale(domain=["mean", "p95"], range=[theme.INK, "#BFB9AC"]),
                legend=alt.Legend(title=None),
            ),
            tooltip=[
                alt.Tooltip("backend:N", title="백엔드"),
                alt.Tooltip("지표:N", title="지표"),
                alt.Tooltip("ms:Q", title="지연시간", format=".2f"),
            ],
        )
    )
