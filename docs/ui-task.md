# UI 작업 지시서

이 문서 하나만 읽고 UI를 완성할 수 있도록 썼다. 코어 라이브러리(`src/vision_dashboard/`)는
이미 구현·검증이 끝났고, **UI 세션은 코어를 수정하지 않는다.**

---

## 1. 프로젝트 개요

`google/vit-base-patch16-224` 로 이미지를 분류하는 Streamlit 대시보드다.
단순 분류 데모가 아니라 **같은 모델을 런타임·정밀도 조합으로 바꿔 끼우며
정확도와 지연시간의 트레이드오프를 보여주는 것**이 이 프로젝트의 정체성이다.

모델 변형 4종:

| key | 라벨 | 런타임 | 정밀도 | 비고 |
|---|---|---|---|---|
| `pytorch_fp32` | PyTorch FP32 | PyTorch | FP32 | 기준선 |
| `pytorch_int8` | PyTorch INT8 (dynamic) | PyTorch | INT8 | Linear 레이어 동적 양자화 |
| `onnx_fp32` | ONNX FP32 | ONNX Runtime | FP32 | 330MB, 로컬 전용 |
| `onnx_int8` | ONNX INT8 (dynamic) | ONNX Runtime | INT8 | 84MB, 배포 기본 백엔드 |

**중요:** 4종이 항상 다 쓸 수 있는 게 아니다. 환경에 따라 가용 여부가 달라지고,
그 판정은 코어가 이미 해준다. UI는 `all_statuses()` 결과를 읽어 렌더링만 하면 된다.
쓸 수 없는 변형은 비활성 상태로 **이유와 함께** 보여줘야 한다 — 숨기지 말 것.
이 폴백 설계를 사용자에게 드러내는 것 자체가 이 프로젝트의 어필 포인트다.

---

## 2. 만들어야 할 파일

```
app.py                              # 엔트리 (얇게 유지, 40줄 내외)
src/vision_dashboard/ui/__init__.py
src/vision_dashboard/ui/state.py    # 세션 상태 · 결과 캐시  ← 가장 중요
src/vision_dashboard/ui/sidebar.py  # 공통 사이드바 (백엔드 선택 · 표시 설정)
src/vision_dashboard/ui/classify.py # 탭 1: 분류
src/vision_dashboard/ui/compare.py  # 탭 2: 백엔드 비교
src/vision_dashboard/ui/bench.py    # 탭 3: 벤치마크
tests/test_app.py                   # Streamlit AppTest 기반 UI 테스트
```

**건드리지 말 것:** `src/vision_dashboard/` 아래의 `config.py`, `backends.py`,
`inference.py`, `labels.py`, `benchmarking.py`, `__init__.py`, 그리고 `scripts/`.
코어에 부족한 게 있으면 고치지 말고 이 문서 맨 아래 "코어에 요청할 사항"에 적어둘 것.

---

## 3. 코어 API 계약

전부 실제 실행으로 검증된 시그니처다.

### `from vision_dashboard import ...`

```python
# --- 백엔드 ---
all_statuses() -> list[VariantStatus]      # 4종 전부, VARIANTS 정의 순서
available_keys() -> list[str]              # 쓸 수 있는 key 만
variant_status(key: str) -> VariantStatus
load_backend(key: str) -> Backend          # lru_cache(maxsize=4). 첫 호출만 느리다
pick_default_variant() -> str | None       # 가벼운 백엔드 우선. 없으면 None

@dataclass(frozen=True)
class VariantStatus:
    key: str; label: str; short: str       # short = "ONNX INT8" 같은 축약 라벨
    runtime: str                           # "pytorch" | "onnx"
    precision: str                         # "fp32" | "int8"
    available: bool
    source: str                            # "local" | "hub" | "pretrained" | "-"
    reason: str                            # 사람이 읽을 사유 (그대로 UI에 노출)
    size_mb: float | None                  # 로컬 파일이 있을 때만 값이 있다
    source_ko: str                         # source 의 한글 표기 (property)

class Backend:
    key: str; label: str
    def infer(self, pixel_values: np.ndarray) -> np.ndarray   # 직접 부를 일 없음

# --- 추론 ---
load_image(source, name: str = "image") -> PIL.Image.Image
    # EXIF 회전 보정 + RGB 변환까지 끝난 이미지를 돌려준다.
    # 실패 시 ImageLoadError 를 던진다. 반드시 try/except 로 감쌀 것.

thumbnail(image, max_pixels: int = 1_200_000) -> PIL.Image.Image
    # 화면 표시용 축소본. st.image 에 넘기는 이미지는 전부 이걸 통과시킬 것.

classify(backend, images: list[Image], names: list[str] | None = None,
         batch_size: int = 8) -> list[ImageResult]
    # 배치 추론. 이미지를 한 장씩 넣지 말고 리스트로 한 번에 넘길 것.

@dataclass
class ImageResult:
    name: str
    probs: np.ndarray        # shape (1000,), 합 1.0
    latency_ms: float        # 이미지당 평균 추론 시간
    def top(self, k: int) -> list[Prediction]
    best: Prediction         # property, top(1)[0]

@dataclass(frozen=True)
class Prediction:
    index: int; label: str; score: float    # score 는 0~1

ImageLoadError                              # load_image 가 던지는 예외

# --- 라벨/이모지 ---
label_of(index: int) -> str
emoji_for(label: str) -> str                # 매칭 실패 시 "🤖"
decorate(label: str, use_emoji: bool) -> str  # "🐱 tabby, tabby cat"

# --- 설정 ---
VARIANTS: dict[str, dict]                   # 순서 있는 정의 (표 렌더링용)
MAX_BATCH_IMAGES = 32                       # 업로드 장수 상한
MODEL_NAME = "google/vit-base-patch16-224"
```

### `from vision_dashboard.benchmarking import ...`

```python
load_results() -> dict | None          # results/benchmark.json. 없으면 None
comparison_rows(payload) -> list[dict] # 기준선 대비 상대 지표까지 계산된 표 데이터
RESULTS_PATH: Path
```

`comparison_rows()` 가 돌려주는 각 행의 키 (그대로 DataFrame 컬럼으로 써도 된다):

```
key, 변형, 용량(MB), 용량 절감(%), 지연 mean(ms), 지연 p95(ms),
속도, Top-1(%), Top-5(%), FP32 일치(%)
```

`용량(MB)` 과 `용량 절감(%)` 은 로컬 파일이 없는 변형에서 `None` 일 수 있다.

`payload` 최상위 구조:

```python
{
  "model": str, "num_eval_images": int, "batch_size": int,
  "environment": {"python", "platform", "processor", "cpu_threads"},
  "variants": { key: {..., "throughput_img_per_sec": float, ...} }
}
```

---

## 4. 절대 지켜야 할 설계 제약

### 4-1. 위젯 조작으로 재추론이 일어나면 안 된다 (최우선)

원본 앱의 가장 큰 결함이 이것이었다. Streamlit은 위젯이 바뀔 때마다 스크립트를
처음부터 다시 실행하는데, 원본은 추론 결과를 저장하지 않아서 **이모지 토글을
껐다 켜기만 해도 업로드한 이미지 전부를 다시 추론**했다.

`ui/state.py` 에 결과 캐시를 두고 다음 규칙을 지킬 것.

- 캐시 키는 `(variant_key, sha1(원본 이미지 바이트))`
- 값은 `ImageResult` (확률 1000개를 통째로 들고 있다)
- **Top-K 개수 변경은 재추론 대상이 아니다.** `ImageResult.top(k)` 를 다시 부르면 끝이다
- **이모지 토글도 재추론 대상이 아니다.** 표시 시점에만 적용한다
- 재추론이 필요한 경우는 단 둘: 새 이미지가 들어왔을 때, 백엔드를 바꿨을 때

권장 구현 형태:

```python
# ui/state.py
def get_or_infer(variant_key: str, items: list[tuple[str, bytes]]) -> list[ImageResult]:
    """캐시 미스인 항목만 모아 한 번에 배치 추론한다."""
    cache = st.session_state.setdefault("results", {})
    missing = [(name, blob) for name, blob in items
               if (variant_key, _digest(blob)) not in cache]
    if missing:
        images = [load_image(io.BytesIO(b), n) for n, b in missing]
        for (name, blob), res in zip(missing, classify(load_backend(variant_key), images,
                                                       [n for n, _ in missing])):
            cache[(variant_key, _digest(blob))] = res
    return [cache[(variant_key, _digest(b))] for _, b in items]
```

미스인 것만 모아 `classify()` 에 **리스트로 한 번에** 넘기는 게 핵심이다.
한 장씩 루프 돌며 부르면 배치 추론 이점이 사라진다.

### 4-2. 예외로 앱이 죽으면 안 된다

`load_image()` 는 `ImageLoadError` 를 던진다. 손상된 파일 하나 때문에 전체가
빨간 에러 화면이 되면 안 된다. 실패한 파일은 `st.warning` 으로 이름과 사유만
알리고 **나머지 파일은 계속 처리**할 것.

### 4-3. 원본 해상도를 브라우저로 보내지 않는다

`st.image()` 에 넘기기 전에 반드시 `thumbnail()` 을 통과시킬 것.

### 4-4. 업로드 장수 상한

`MAX_BATCH_IMAGES`(32) 를 넘으면 앞에서부터 32장만 처리하고, 몇 장을 잘랐는지
`st.warning` 으로 알릴 것. 조용히 자르지 말 것.

### 4-5. 백엔드 로딩은 눈에 보이게

`load_backend()` 첫 호출은 수십 초가 걸린다(측정값: PyTorch FP32 최초 로드 38.6초).
반드시 `st.spinner` 로 감싸고, 무엇을 하는 중인지 쓸 것.

---

## 5. 화면 구성

### 공통 사이드바 (`ui/sidebar.py`)

- **백엔드 선택** — `all_statuses()` 를 순회. 사용 가능한 것만 `st.radio`/`st.selectbox`
  선택지로 넣고, 불가능한 것은 아래에 `st.caption` 으로 `label — reason` 표시.
  기본 선택값은 `pick_default_variant()`.
  선택된 변형의 `source_ko`, `size_mb` 를 캡션으로 보여줄 것 (예: "로컬 파일 · 83.7 MB").
- **표시 설정** — 이모지 토글, Top-K 슬라이더(1~10, 기본 5).
  4-1 규칙에 따라 **이 둘은 재추론을 유발하지 않아야 한다.**
- 하단에 모델명(`MODEL_NAME`)과 스레드 수 캡션.

사용 가능한 백엔드가 하나도 없으면(전 변형 `available=False`) 사이드바 대신
`st.error` 로 `scripts/export_models.py` 실행을 안내하고 조기 반환할 것.

### 탭 1: 분류 (`ui/classify.py`)

원본 앱의 기능을 계승하되 결함을 고친 화면이다.

- 입력: 파일 업로드(다중, jpg/png/jpeg) / 카메라 촬영 두 모드
- **업로드한 이미지 전부를 썸네일 그리드로 미리보기** (원본은 대표 1장만 보여줬다)
- '분석 실행' 버튼 → `get_or_infer()` 호출
- 결과 카드: 썸네일 + Top-1 라벨(이모지 적용) + 신뢰도 + 이미지당 지연시간(ms)
- **Top-K 막대 차트** — Altair 가로 막대. 원본 README가 "Bar Chart로 시각화"라고
  써놓고 실제로는 없었던 부분이다. 반드시 넣을 것
- 결과 내보내기: CSV / JSON 다운로드 버튼
  (컬럼: `image, rank, label, score, variant, latency_ms`)
- 진행률 표시는 배치 단위로. 원본처럼 첫 이미지 내내 0%에 멈춰 있으면 안 된다

### 탭 2: 백엔드 비교 (`ui/compare.py`)

**이 프로젝트의 차별화 지점이다. 가장 공들일 것.**

- 이미지 1장 입력 + 비교할 백엔드 다중 선택(`st.multiselect`, 기본 = 사용 가능한 전부)
- 선택한 백엔드마다 추론 → 나란히 비교:
  - Top-1 예측 라벨과 신뢰도
  - 이미지당 지연시간(ms)
  - **기준선(가장 무거운 FP32) 대비 속도 배율**
  - **예측 일치 여부** — 기준선과 Top-1이 같으면 ✅, 다르면 ⚠️
- 지연시간 막대 차트 + Top-5 확률 분포를 백엔드별로 겹쳐 비교
- 한 줄 요약을 강조 표시할 것.
  예: "ONNX INT8은 FP32 대비 1.8배 빠르고 용량은 74.7% 작으면서 같은 답을 냈다"
  — 수치는 반드시 측정값에서 계산할 것. 하드코딩 금지

### 탭 3: 벤치마크 (`ui/bench.py`)

- `load_results()` 로 `results/benchmark.json` 을 읽어 렌더링
- **결과 파일이 없으면**(`None`) `st.info` 로 `python scripts/run_benchmark.py --n 200`
  실행을 안내하고 종료. 이 분기를 빠뜨리지 말 것 — 클론 직후엔 파일이 없다
- `comparison_rows()` 를 `st.dataframe` 으로 표시
- 차트 3종: 용량 비교, 지연시간(mean/p95) 비교, Top-1 정확도 비교
- 측정 환경(`payload["environment"]`)을 캡션으로 명시.
  다른 CPU에서 잰 수치는 비교 의미가 없으므로 이 정보가 표에 붙어 있어야 한다
- 앱 내 재측정 버튼은 **선택 구현**. 수 분이 걸려 Streamlit 세션에 부담이 크다.
  만든다면 반드시 이미지 수를 줄인 축약 모드(예: 20장)로 할 것

---

## 6. `app.py` 골격

```python
"""ViT 이미지 분류 대시보드 엔트리."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

# set_page_config 는 다른 어떤 st 명령보다 먼저 와야 한다.
# (원본 app.py 는 st.spinner 뒤에 있었다. 지금 버전에서 예외가 나진 않지만 고칠 것)
st.set_page_config(page_title="ViT Vision Dashboard", page_icon="🖼️", layout="wide")

from vision_dashboard.ui import bench, classify, compare, sidebar  # noqa: E402

def main() -> None:
    st.title("🖼️ ViT 이미지 분류 대시보드")
    st.caption("같은 모델을 런타임·정밀도별로 바꿔 끼우며 정확도와 속도를 비교한다")

    settings = sidebar.render()
    if settings is None:      # 사용 가능한 백엔드가 없는 경우
        return

    tab1, tab2, tab3 = st.tabs(["🔍 분류", "⚖️ 백엔드 비교", "📊 벤치마크"])
    with tab1:
        classify.render(settings)
    with tab2:
        compare.render(settings)
    with tab3:
        bench.render()

main()
```

`sidebar.render()` 는 선택된 백엔드 key, 이모지 여부, Top-K 를 담은 작은
dataclass 나 dict 를 돌려주면 된다. 형태는 자유.

---

## 7. 테스트 (`tests/test_app.py`)

Streamlit 공식 `AppTest` 를 쓴다. 실제 추론은 느리므로 **백엔드는 반드시 목으로 대체**할 것.

```python
from streamlit.testing.v1 import AppTest

def test_app_runs_without_exception():
    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert not at.exception
```

최소한 다음 3가지는 검증할 것.

1. 앱이 예외 없이 뜬다
2. 사용 가능한 백엔드가 없을 때 안내 메시지가 나오고 죽지 않는다
3. **Top-K 슬라이더를 움직여도 추론 함수가 다시 호출되지 않는다**
   (`classify` 를 monkeypatch 로 카운터를 붙여 호출 횟수를 센다 — 4-1 규칙의 회귀 방지)

3번이 이 테스트 묶음의 핵심이다. 원본에서 실제로 깨져 있던 동작이다.

---

## 8. 개발 환경

```bash
# 가상환경은 이미 .venv 에 준비돼 있다 (Python 3.12)
.venv/bin/streamlit run app.py

# 테스트
.venv/bin/pytest -q
```

주요 버전: `streamlit 1.61.1`, `torch 2.13.0`, `transformers 5.14.1`,
`onnxruntime 1.28.0`, `numpy 2.5.1`, `pandas 3.0.5`, `altair 6.2.2`.

**transformers 는 5.x 다.** 4.x 시절 예제 코드를 그대로 붙여넣지 말 것.

현재 상태에서 `pytorch_fp32` / `pytorch_int8` 는 바로 쓸 수 있다(원본에서 즉석 구성).
ONNX 변형을 쓰려면 `python scripts/export_models.py` 를 먼저 돌려야 한다.

---

## 9. 하지 말 것

- 코어 모듈 수정 (`config.py`, `backends.py`, `inference.py`, `labels.py`, `benchmarking.py`)
- `transformers.pipeline()` 직접 호출 — 백엔드 추상화를 우회하게 된다
- 이모지 매핑을 UI에서 다시 구현 — `emoji_for()` / `decorate()` 를 쓸 것
- 하드코딩된 성능 수치를 화면에 표시 — 전부 측정값에서 계산할 것
- 결과를 세션에 저장하지 않고 매 rerun 마다 추론 (4-1 위반)
- `requirements.txt` / `pyproject.toml` 수정

---

## 10. 코어에 요청할 사항

UI를 만들다 코어에 부족한 게 발견되면 직접 고치지 말고 여기에 적어둘 것.

- ~~`classify()` 에 진행률 콜백이 없다. 배치 단위 진행률을 내려면 UI 쪽에서
  `INFER_BATCH_SIZE` 만큼 잘라 여러 번 호출해야 한다 (`ui/state.py::_infer_missing`).
  배치 이점은 유지되지만, `on_batch: Callable[[int, int], None]` 같은 인자가 있으면
  UI 가 청크 분할을 몰라도 된다. 급하지는 않다.~~

  **처리 완료.** `classify()` 와 `infer_batched()` 에 `on_batch` 를 추가했다.

  ```python
  classify(backend, images, names=None, batch_size=8,
           on_batch: Callable[[int, int], None] | None = None) -> list[ImageResult]
      # 배치가 끝날 때마다 (처리한 장수, 전체 장수) 로 호출한다.
      # 콜백에 걸린 시간은 latency_ms 집계에서 제외한다 — UI 를 그리는 시간이
      # 추론 지연시간으로 잡히면 안 되기 때문이다.
  ```

  `ui/state.py::_infer_missing` 의 청크 루프는 제거했고, 이제 `classify()` 를
  한 번만 부른다. 회귀 방지 테스트 3개를 `tests/test_inference.py` 에 추가했다
  (콜백 호출 시퀀스, 콜백 없이도 동작, 콜백 시간이 지연시간에 섞이지 않음).
