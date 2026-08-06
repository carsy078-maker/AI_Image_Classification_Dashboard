"""라벨 조회와 이모지 매핑.

네트워크나 모델 가중치 없이 도는 테스트다. assets/imagenet_classes.json 만 있으면 된다.
"""

from vision_dashboard import labels


def test_id2label_has_1000_classes():
    assert len(labels.id2label()) == 1000


def test_known_class_indices():
    # ImageNet-1k 표준 인덱스. 뒤바뀌면 정확도 계산이 통째로 틀어진다.
    assert labels.label_of(0).startswith("tench")
    assert "tabby" in labels.label_of(281)


def test_label_of_out_of_range_does_not_raise():
    assert labels.label_of(9999) == "class_9999"
    assert labels.label_of(-1) == "class_-1"


def test_emoji_matches_expected_category():
    assert labels.emoji_for("golden retriever") == "🐶"
    assert labels.emoji_for("tabby, tabby cat") == "🐱"
    assert labels.emoji_for("Egyptian cat") == "🐱"
    assert labels.emoji_for("airliner") == "✈️"


def test_food_rules_win_over_animal_rules():
    """'hot dog' 는 개가 아니다. 규칙 순서가 뒤집히면 여기서 잡힌다."""
    assert labels.emoji_for("hotdog, hot dog, red hot") == "🍔"


def test_unknown_label_falls_back():
    assert labels.emoji_for("nonexistent gizmo") == labels.FALLBACK_EMOJI


def test_decorate_respects_toggle():
    assert labels.decorate("Egyptian cat", use_emoji=False) == "Egyptian cat"
    assert labels.decorate("Egyptian cat", use_emoji=True).startswith("🐱 ")


def test_emoji_coverage_is_meaningfully_high():
    """원본은 if 문 8개로 8클래스만 덮었다. 규칙을 줄이면 여기서 걸린다."""
    assert labels.emoji_coverage() >= 55.0
