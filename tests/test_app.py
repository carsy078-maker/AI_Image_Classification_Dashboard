"""Streamlit UI 테스트.

실제 추론은 느리므로 백엔드와 `classify()` 는 목으로 대체한다.
이 묶음의 핵심은 "표시 설정을 바꿔도 추론이 다시 돌지 않는다" 는 회귀 방지다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from vision_dashboard import ImageResult, VariantStatus
from vision_dashboard.ui import classify as classify_ui
from vision_dashboard.ui import sidebar as sidebar_ui
from vision_dashboard.ui import state as state_module

# AppTest 는 상대 경로를 호출한 파일 기준으로 푼다.
APP = str(Path(__file__).resolve().parents[1] / "app.py")
TIMEOUT = 90


def _fake_result(name: str) -> ImageResult:
    """확률 1000개를 갖춘 결과. 281번(tabby cat)이 최상위가 되게 만든다."""
    probs = np.full(1000, 0.0001, dtype=np.float64)
    probs[281] = 0.82
    probs[282] = 0.11
    probs /= probs.sum()
    return ImageResult(name=name, probs=probs, latency_ms=42.5)


@pytest.fixture
def stub_inference(monkeypatch, sample_image_bytes):
    """추론을 목으로 바꾸고 호출 횟수를 센다.

    `state` 모듈이 import 한 이름을 갈아끼워야 한다. 코어 모듈을 패치하면
    이미 바인딩된 참조가 남아 목이 걸리지 않는다.
    """
    calls: list[int] = []

    def fake_classify(backend, images, names=None, batch_size=8, on_batch=None):
        calls.append(len(images))
        names = names or [f"image_{i + 1}" for i in range(len(images))]
        # 코어가 배치마다 부르는 진행률 콜백. UI 가 이걸로 막대를 움직인다.
        if on_batch:
            on_batch(len(images), len(images))
        return [_fake_result(name) for name in names]

    monkeypatch.setattr(state_module, "classify", fake_classify)
    monkeypatch.setattr(state_module, "load_backend", lambda key: object())
    monkeypatch.setattr(classify_ui, "_collect_input", lambda: [("cat.jpg", sample_image_bytes)])
    return calls


def test_app_runs_without_exception():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    assert not at.exception
    assert at.title[0].value == "ViT 이미지 분류 대시보드"


def test_no_backend_shows_guidance_instead_of_crashing(monkeypatch):
    """전 변형이 사용 불가일 때 안내를 띄우고 조용히 끝나야 한다."""
    blocked = [
        VariantStatus(
            key=key,
            label=key,
            short=key,
            runtime="pytorch",
            precision="fp32",
            available=False,
            source="-",
            reason="scripts/export_models.py 를 실행해 가중치를 만들어야 한다",
            size_mb=None,
        )
        for key in ("pytorch_fp32", "onnx_int8")
    ]
    monkeypatch.setattr(sidebar_ui, "all_statuses", lambda: blocked)

    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()

    assert not at.exception
    assert at.error, "사용 가능한 백엔드가 없으면 안내가 떠야 한다"
    assert "export_models.py" in at.error[0].value


def test_topk_slider_does_not_retrigger_inference(stub_inference):
    """Top-K 를 바꾸는 것은 표시 문제다. 추론이 다시 돌면 안 된다."""
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    assert not at.exception
    assert stub_inference == [], "버튼을 누르기 전에는 추론하지 않는다"

    at.button(key="vd_run_classify").click().run()
    assert not at.exception
    assert stub_inference == [1], "분석 실행에서 한 번만 추론해야 한다"

    at.slider(key="vd_topk").set_value(9).run()
    assert not at.exception
    assert stub_inference == [1], "Top-K 변경으로 추론이 다시 돌면 안 된다"

    at.toggle(key="vd_emoji").set_value(False).run()
    assert not at.exception
    assert stub_inference == [1], "이모지 토글로 추론이 다시 돌면 안 된다"


def test_results_survive_rerun(stub_inference):
    """캐시된 결과는 위젯을 조작한 뒤에도 화면에 그대로 남아 있어야 한다."""
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.button(key="vd_run_classify").click().run()

    def shows_prediction(app: AppTest) -> bool:
        return any("tabby" in block.value for block in app.markdown)

    assert shows_prediction(at), "분석 직후 예측 라벨이 보여야 한다"

    at.slider(key="vd_topk").set_value(3).run()
    assert not at.exception
    assert shows_prediction(at), "Top-K 를 바꿔도 결과는 남아 있어야 한다"
    assert stub_inference == [1]
