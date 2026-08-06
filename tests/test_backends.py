"""가중치 확보 경로 판정.

환경에 따라 쓸 수 있는 변형이 달라지는 게 이 프로젝트의 핵심 제약이라,
판정 로직은 모델 없이도 검증할 수 있어야 한다.
"""

import pytest

from vision_dashboard import backends
from vision_dashboard.config import VARIANTS


@pytest.fixture(autouse=True)
def _clear_backend_cache():
    backends.load_backend.cache_clear()
    yield
    backends.load_backend.cache_clear()


def _point_all_variants_at(tmp_path, monkeypatch):
    """모든 변형의 가중치 경로를 빈 임시 디렉터리로 돌린다."""
    for key, meta in VARIANTS.items():
        monkeypatch.setitem(meta, "path", tmp_path / f"{key}.bin")


def test_all_statuses_covers_every_variant():
    assert [s.key for s in backends.all_statuses()] == list(VARIANTS)


def test_local_file_is_preferred(tmp_path, monkeypatch):
    _point_all_variants_at(tmp_path, monkeypatch)
    path = VARIANTS["onnx_int8"]["path"]
    path.write_bytes(b"x" * 2048)

    status = backends.variant_status("onnx_int8")
    assert status.available
    assert status.source == "local"
    assert status.size_mb == pytest.approx(0.0, abs=0.01)


def test_hub_fallback_when_repo_configured(tmp_path, monkeypatch):
    """배포 환경 시나리오: 로컬 파일은 없고 Hub 저장소만 설정된 경우."""
    _point_all_variants_at(tmp_path, monkeypatch)
    monkeypatch.setattr(backends, "HUB_ONNX_REPO", "someone/vit-onnx")

    status = backends.variant_status("onnx_int8")
    assert status.available
    assert status.source == "hub"


def test_pytorch_variants_work_without_any_file(tmp_path, monkeypatch):
    """PyTorch 쪽은 원본에서 즉석 구성할 수 있어 파일이 없어도 쓸 수 있다."""
    _point_all_variants_at(tmp_path, monkeypatch)
    for key in ("pytorch_fp32", "pytorch_int8"):
        status = backends.variant_status(key)
        assert status.available
        assert status.source == "pretrained"


def test_onnx_fp32_is_unavailable_without_local_file(tmp_path, monkeypatch):
    """330MB 라 Hub 배포 대상이 아니다. 로컬 export 없이는 못 쓴다."""
    _point_all_variants_at(tmp_path, monkeypatch)
    monkeypatch.setattr(backends, "HUB_ONNX_REPO", "")

    status = backends.variant_status("onnx_fp32")
    assert not status.available
    assert "export_models" in status.reason


def test_unavailable_variants_still_expose_a_reason(tmp_path, monkeypatch):
    """UI 가 '왜 못 쓰는지' 를 그대로 보여줄 수 있어야 한다."""
    _point_all_variants_at(tmp_path, monkeypatch)
    monkeypatch.setattr(backends, "HUB_ONNX_REPO", "")
    for status in backends.all_statuses():
        assert status.reason


def test_available_keys_matches_statuses():
    expected = [s.key for s in backends.all_statuses() if s.available]
    assert backends.available_keys() == expected


def test_default_variant_prefers_lighter_backend(tmp_path, monkeypatch):
    """ONNX INT8 이 쓸 수 있으면 그걸 고른다 (가장 가볍다)."""
    _point_all_variants_at(tmp_path, monkeypatch)
    VARIANTS["onnx_int8"]["path"].write_bytes(b"x")
    assert backends.pick_default_variant() == "onnx_int8"


def test_default_variant_falls_back_to_pytorch(tmp_path, monkeypatch):
    _point_all_variants_at(tmp_path, monkeypatch)
    monkeypatch.setattr(backends, "HUB_ONNX_REPO", "")
    assert backends.pick_default_variant() == "pytorch_int8"


def test_resolve_weight_path_raises_for_missing_required_file(tmp_path, monkeypatch):
    _point_all_variants_at(tmp_path, monkeypatch)
    monkeypatch.setattr(backends, "HUB_ONNX_REPO", "")
    with pytest.raises(FileNotFoundError, match="export_models"):
        backends.resolve_weight_path("onnx_fp32")


def test_resolve_weight_path_returns_none_for_optional_file(tmp_path, monkeypatch):
    _point_all_variants_at(tmp_path, monkeypatch)
    assert backends.resolve_weight_path("pytorch_fp32") is None


def test_load_backend_rejects_unknown_key():
    with pytest.raises(KeyError):
        backends.load_backend("nope")


def test_quantization_engine_is_resolvable():
    """torch 기본값은 'none' 이라, 명시하지 않으면 양자화가 런타임 에러를 낸다."""
    assert backends.setup_quantization_engine() in {"fbgemm", "qnnpack", "x86", "onednn"}
