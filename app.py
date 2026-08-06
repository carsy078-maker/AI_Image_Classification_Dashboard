"""ViT 이미지 분류 대시보드 엔트리."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

# set_page_config 는 다른 어떤 st 명령보다 먼저 와야 한다.
st.set_page_config(
    page_title="ViT Vision Dashboard",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from vision_dashboard.ui import bench, classify, compare, sidebar, theme  # noqa: E402


def main() -> None:
    theme.inject()

    st.title("ViT 이미지 분류 대시보드")
    st.caption("같은 모델을 런타임·정밀도별로 바꿔 끼우며 정확도와 속도를 비교한다")

    settings = sidebar.render()
    if settings is None:  # 쓸 수 있는 백엔드가 없는 경우
        return

    classify_tab, compare_tab, bench_tab = st.tabs(["분류", "백엔드 비교", "벤치마크"])
    with classify_tab:
        classify.render(settings)
    with compare_tab:
        compare.render(settings)
    with bench_tab:
        bench.render()


main()
