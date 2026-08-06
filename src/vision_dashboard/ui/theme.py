"""디자인 토큰과 공통 렌더링 조각.

색·간격·타이포는 전부 여기서만 정의한다. 다른 UI 모듈은 상수와 헬퍼만 가져다 쓰고
직접 색값을 적지 않는다.

톤: 웜 모노크롬 에디토리얼. 색은 의미가 있을 때만(일치/불일치/사용불가) 쓰고,
나머지 위계는 타이포 크기와 여백으로 만든다.
"""

from __future__ import annotations

import html

import altair as alt
import streamlit as st

# --- 색 토큰 ---------------------------------------------------------------
INK = "#1F1E1C"  # 본문. 순수 검정은 쓰지 않는다
MUTED = "#78766F"  # 보조 설명
FAINT = "#A5A29A"  # 캡션, 비활성
LINE = "#E5E3DE"  # 모든 테두리와 구분선
CANVAS = "#FBFBFA"
SURFACE = "#FFFFFF"
SUBTLE = "#F4F3F0"

# 의미색은 채도를 낮춘 파스텔 배경 + 진한 글자 조합으로만 쓴다.
TONES = {
    "positive": ("#EDF3EC", "#346538"),
    "caution": ("#FBF3DB", "#8A5D00"),
    "negative": ("#FDEBEC", "#9F2F2D"),
    "info": ("#E7F0F7", "#1F6C9F"),
    "neutral": ("#F0EFEB", "#5C5A54"),
}

# 백엔드 4종 구분색. 런타임이 색 계열을 가르고, 정밀도가 명도를 가른다.
VARIANT_COLORS = {
    "pytorch_fp32": "#2F3437",
    "pytorch_int8": "#A8A093",
    "onnx_fp32": "#4A6FA5",
    "onnx_int8": "#6E8B6F",
}

SANS = (
    "'Pretendard Variable', Pretendard, -apple-system, BlinkMacSystemFont, "
    "'Apple SD Gothic Neo', 'Segoe UI', system-ui, sans-serif"
)
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', 'JetBrains Mono', Menlo, monospace"


def variant_color(key: str) -> str:
    return VARIANT_COLORS.get(key, INK)


# --- 전역 스타일 -----------------------------------------------------------

_CSS = f"""
<style>
/* 기본 상단 여백이 과해서 제목이 화면 중앙까지 밀린다 */
[data-testid="stMainBlockContainer"] {{
    padding-top: 2.75rem;
    padding-bottom: 5rem;
    max-width: 1180px;
}}

/* 숫자는 자리폭을 고정해야 값이 바뀔 때 열이 흔들리지 않는다 */
[data-testid="stMetricValue"], .vd-num, table, [data-testid="stDataFrame"] {{
    font-variant-numeric: tabular-nums;
    font-feature-settings: "tnum";
}}

[data-testid="stMetricLabel"] p {{
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: {MUTED};
    font-weight: 560;
}}
[data-testid="stMetricValue"] {{
    letter-spacing: -0.02em;
    line-height: 1.15;
}}

/* 탭 — 박스가 아니라 밑줄로 구분하는 에디토리얼 방식 */
.stTabs [data-baseweb="tab-list"] {{
    gap: 2rem;
    border-bottom: 1px solid {LINE};
}}
.stTabs [data-baseweb="tab"] {{
    padding: 0.35rem 0 0.7rem 0;
    font-weight: 560;
    letter-spacing: -0.005em;
    color: {MUTED};
}}
.stTabs [aria-selected="true"] {{ color: {INK}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {INK}; height: 2px; }}
.stTabs [data-baseweb="tab-border"] {{ display: none; }}

/* 카드는 1px 헤어라인만. 그림자는 hover 에서만 아주 옅게 */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {{
    transition: box-shadow 200ms cubic-bezier(0.16, 1, 0.3, 1);
}}

hr {{ border-color: {LINE}; margin: 1.5rem 0; }}

/* 사이드바 라디오를 목록처럼 붙여 읽히게 한다 */
[data-testid="stSidebar"] [role="radiogroup"] {{ gap: 0.15rem; }}

/* --- 커스텀 조각 --- */
.vd-eyebrow {{
    font-size: 0.68rem;
    font-weight: 580;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {FAINT};
    margin: 0 0 0.4rem 0;
}}
.vd-pill {{
    display: inline-block;
    padding: 0.12rem 0.55rem;
    border-radius: 9999px;
    font-size: 0.7rem;
    font-weight: 580;
    letter-spacing: 0.02em;
    white-space: nowrap;
    vertical-align: middle;
}}
.vd-mono {{ font-family: {MONO}; font-variant-numeric: tabular-nums; }}

/* 백엔드 비교용 — 하나의 테두리 안을 세로선으로 나눈 지표 띠 */
.vd-strip {{
    display: grid;
    border: 1px solid {LINE};
    border-radius: 10px;
    background: {SURFACE};
    overflow: hidden;
}}
.vd-strip > .vd-cell {{
    padding: 1.15rem 1.35rem;
    border-left: 1px solid {LINE};
    min-width: 0;
}}
.vd-strip > .vd-cell:first-child {{ border-left: none; }}
.vd-cell-label {{
    font-size: 0.72rem;
    font-weight: 560;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: {MUTED};
    margin-bottom: 0.5rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
.vd-cell-value {{
    font-size: 1.6rem;
    font-weight: 640;
    letter-spacing: -0.025em;
    line-height: 1.1;
    color: {INK};
    font-variant-numeric: tabular-nums;
}}
.vd-cell-value .vd-unit {{
    font-size: 0.85rem;
    font-weight: 500;
    color: {FAINT};
    letter-spacing: 0;
    margin-left: 0.2rem;
}}
.vd-cell-foot {{
    margin-top: 0.55rem;
    padding-top: 0.55rem;
    border-top: 1px solid {LINE};
    font-size: 0.76rem;
    color: {MUTED};
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}

/* 결론 한 줄 — 페이지에서 가장 크게 읽혀야 하는 문장 */
.vd-verdict {{
    border: 1px solid {LINE};
    border-left: 3px solid {INK};
    border-radius: 8px;
    background: {SURFACE};
    padding: 1.1rem 1.3rem;
    font-size: 1.02rem;
    line-height: 1.6;
    color: {INK};
    letter-spacing: -0.01em;
}}
.vd-verdict strong {{ font-weight: 640; }}

/* 사용할 수 없는 백엔드 — 숨기지 않고 사유와 함께 흐리게 보여준다 */
.vd-unavail {{
    font-size: 0.78rem;
    color: {FAINT};
    line-height: 1.65;
    padding: 0.1rem 0;
}}
.vd-unavail .vd-name {{ color: {MUTED}; font-weight: 560; }}
</style>
"""


def inject() -> None:
    """전역 CSS 를 한 번만 심는다."""
    st.html(_CSS)
    _register_altair_theme()


# --- Altair -----------------------------------------------------------------


def _register_altair_theme() -> None:
    """차트도 같은 타이포·선 색을 쓰게 한다. 등록은 프로세스당 한 번이면 된다."""
    if getattr(_register_altair_theme, "_done", False):
        return

    @alt.theme.register("vision_dashboard", enable=True)
    def _theme() -> alt.theme.ThemeConfig:
        return {
            "config": {
                "background": "transparent",
                "font": SANS,
                "view": {"stroke": "transparent", "continuousHeight": 240},
                "arc": {"stroke": SURFACE},
                "axis": {
                    "labelColor": MUTED,
                    "labelFontSize": 11,
                    "titleColor": MUTED,
                    "titleFontSize": 11,
                    "titleFontWeight": 500,
                    "titlePadding": 10,
                    "domainColor": LINE,
                    "tickColor": LINE,
                    "tickSize": 4,
                    "grid": False,
                    "gridColor": LINE,
                    "gridDash": [2, 3],
                },
                "axisX": {"labelAngle": 0},
                "axisY": {"grid": True, "domain": False, "ticks": False, "labelPadding": 6},
                "legend": {
                    "labelColor": INK,
                    "labelFontSize": 11,
                    "titleColor": MUTED,
                    "titleFontSize": 11,
                    "titleFontWeight": 500,
                    "symbolType": "square",
                    "symbolSize": 90,
                    "orient": "top",
                    "direction": "horizontal",
                    "offset": 6,
                },
                "title": {
                    "color": INK,
                    "fontSize": 13,
                    "fontWeight": 600,
                    "anchor": "start",
                    "offset": 12,
                    "subtitleColor": MUTED,
                    "subtitleFontSize": 11,
                },
                "bar": {"cornerRadiusEnd": 3},
                "text": {"color": INK, "fontSize": 11},
                "range": {"category": list(VARIANT_COLORS.values())},
            }
        }

    _register_altair_theme._done = True  # type: ignore[attr-defined]


def chart(spec: alt.Chart, height: int | None = None) -> None:
    """차트를 컨테이너 폭에 맞춰 그린다. Streamlit 기본 테마는 끄고 위 테마를 쓴다."""
    if height is not None:
        spec = spec.properties(height=height)
    st.altair_chart(spec, width="stretch", theme=None)


# --- HTML 조각 ---------------------------------------------------------------
# 사용자 입력(파일명)과 모델 출력(라벨)이 그대로 들어오므로 전부 escape 한다.


def esc(value: object) -> str:
    """HTML 조각에 값을 넣기 전에 반드시 통과시킨다."""
    return html.escape(str(value))


_esc = esc


def eyebrow(text: str) -> str:
    return f'<p class="vd-eyebrow">{_esc(text)}</p>'


def pill(text: str, tone: str = "neutral") -> str:
    bg, fg = TONES.get(tone, TONES["neutral"])
    return f'<span class="vd-pill" style="background:{bg};color:{fg}">{_esc(text)}</span>'


def section(title: str, description: str | None = None) -> None:
    """섹션 제목. 제목과 설명 사이 간격을 일정하게 유지하려고 함수로 묶었다."""
    st.markdown(f"##### {_esc(title)}", unsafe_allow_html=False)
    if description:
        st.caption(description)


def stat_strip(cells: list[dict]) -> None:
    """지표 띠. 테두리 하나 안을 세로선으로 나눠 값들을 나란히 놓는다.

    cells 의 각 항목: label(필수), value(필수), unit, foot, tone, color
    """
    if not cells:
        return
    blocks = []
    for cell in cells:
        unit = f'<span class="vd-unit">{_esc(cell["unit"])}</span>' if cell.get("unit") else ""
        color = cell.get("color") or INK
        foot = f'<div class="vd-cell-foot">{cell["foot"]}</div>' if cell.get("foot") else ""
        blocks.append(
            f'<div class="vd-cell">'
            f'<div class="vd-cell-label">{_esc(cell["label"])}</div>'
            f'<div class="vd-cell-value" style="color:{color}">{_esc(cell["value"])}{unit}</div>'
            f"{foot}</div>"
        )
    cols = f"grid-template-columns: repeat({len(cells)}, minmax(0, 1fr));"
    st.html(f'<div class="vd-strip" style="{cols}">{"".join(blocks)}</div>')


def verdict(text_html: str) -> None:
    """측정값에서 계산한 결론 한 줄. 인자는 이미 escape 된 HTML 이어야 한다."""
    st.html(f'<div class="vd-verdict">{text_html}</div>')


def unavailable_list(items: list[tuple[str, str]]) -> None:
    """(이름, 사유) 목록을 흐린 캡션으로. 쓸 수 없는 백엔드를 숨기지 않기 위한 것."""
    rows = "".join(
        f'<div class="vd-unavail"><span class="vd-name">{_esc(name)}</span> — {_esc(reason)}</div>'
        for name, reason in items
    )
    st.html(rows)
