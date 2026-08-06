# ViT Image Classification Dashboard

**라이브 데모 → https://vit.lucenta.duckdns.org**
(OCI Ampere A1 / Ubuntu 24.04 aarch64 / Docker. 4종 백엔드 모두 사용 가능)

사진을 올리면 **ImageNet-1k 의 1000개 클래스 중 무엇인지 맞히는** 웹 대시보드다.
Vision Transformer(`google/vit-base-patch16-224`)로 추론하고, 어떤 라벨에 얼마나
확신하는지를 Top-K 확률로 함께 보여준다.

여기에 더해, **같은 모델을 런타임(PyTorch / ONNX Runtime)과 정밀도(FP32 / INT8)
조합으로 바꿔 끼우며** 정확도와 속도가 어떻게 달라지는지 직접 비교할 수 있다.

![분류 화면](assets/screenshots/01-classify.png)

> 네 장을 한 번에 올린 결과. 각 이미지마다 Top-1 라벨과 신뢰도, Top-5 확률 분포,
> 추론에 걸린 시간을 함께 낸다.

---

## 무엇을 분류하는가

**모델** — `google/vit-base-patch16-224`. 224×224 이미지를 16×16 패치 196개로
쪼개 Transformer 로 처리하는 ViT-Base (86M 파라미터). ImageNet-21k 로 사전학습한 뒤
ImageNet-1k 로 파인튜닝된 가중치를 그대로 쓴다 (추가 학습은 하지 않는다).

**분류 대상** — ImageNet-1k 의 1000개 클래스. 인덱스가 의미 순으로 정렬돼 있어서
**0~397 번(398개)이 동물**이고 그중 **151~268 번(118종)이 개 품종**이다. 398 번부터가
사물·음식·식물·탈것·장소다. 라벨 목록은
[`assets/imagenet_classes.json`](assets/imagenet_classes.json) 에 캐시해 둔다.

이 편중은 평가 설계에도 영향을 준다 — 앞에서 200장을 자르면 사실상 어류·조류
분류기를 평가하게 되므로, [균등 간격으로 추출](#평가-표본-추출)한다.

실제 예측 예시 (ONNX INT8):

| 입력 | Top-1 예측 | 신뢰도 |
|---|---|---:|
| 개 사진 | 🐶 `flat-coated retriever` | 90.1% |
| 얼룩말 사진 | 🦓 `zebra` | 99.7% |
| 커피머신 사진 | ☕ `espresso maker` | 98.9% |
| 데이지 사진 | 🌸 `daisy` | 99.7% |
| 고양이 사진 | 🐱 `tabby, tabby cat` | 66.3% |

라벨 앞의 이모지는 키워드 규칙 테이블로 붙인다. 1000개 클래스 중 **616개(61.6%)**
가 매핑되고, 나머지는 🤖 로 떨어진다. 규칙은 [`labels.py`](src/vision_dashboard/labels.py)
의 데이터 테이블이라 추가가 쉽다.

---

## 분류 파이프라인

업로드부터 화면에 뜨기까지 거치는 단계다. 전부
[`inference.py`](src/vision_dashboard/inference.py) 에 있다.

```
업로드 바이트
   │
   ├─ 1. 이미지 열기 + EXIF 회전 보정
   │     휴대폰 사진은 세로로 찍어도 바이트는 가로로 저장되고 방향은 EXIF 태그에만
   │     들어있다. 보정하지 않으면 눕혀진 이미지를 추론하게 된다.
   │
   ├─ 2. RGB 변환
   │     RGBA(투명 PNG)·그레이스케일·팔레트 이미지를 3채널로 통일한다.
   │
   ├─ 3. 224×224 리사이즈 + 정규화
   │     mean=std=0.5 로 [-1, 1] 범위에 맞춘다. ViT 가 학습된 전처리와
   │     동일해야 하므로 transformers 의 프로세서 설정을 그대로 따른다.
   │
   ├─ 4. 배치 추론 → (N, 1000) logits
   │     한 장씩이 아니라 8장 단위로 묶어 넣는다.
   │
   ├─ 5. softmax → 확률 1000개
   │     오버플로를 막으려 max 를 빼고 지수화한다.
   │
   └─ 6. Top-K 추출 + 이모지 부착 → 화면
         확률 1000개를 통째로 들고 있으므로 K 를 바꿔도 다시 추론하지 않는다.
```

---

## 분류 품질

ImageNet-1k 에서 균등 추출한 200장으로 측정했다 (ONNX INT8 기준).

| 지표 | 값 |
|---|---:|
| Top-1 정확도 | **89.5%** (179/200) |
| Top-5 정확도 | **99.0%** |
| Top-1 을 틀렸지만 Top-5 안에 정답이 있음 | 19장 |
| Top-5 에도 정답이 없음 | **2장** |

### 틀릴 때는 이렇게 틀린다

오답 21장을 들여다보면 **엉뚱한 답이 아니라 같은 범주 안의 비슷한 종**으로
쏠린다. 그래서 Top-5 정확도가 99%까지 올라간다.

| 정답 | 예측 | 신뢰도 |
|---|---|---:|
| `Appenzeller` | `Bernese mountain dog` | 97.7% |
| `Brittany spaniel` | `Welsh springer spaniel` | 84.9% |
| `green snake` | `green mamba` | 91.0% |
| `jaguar` | `cougar, puma` | 99.7% |
| `mantis` | `lacewing fly` | 27.7% |

스위스 산악견끼리, 스패니얼끼리, 녹색 뱀끼리 헷갈린다. 사람도 사진만 보고
구분하기 어려운 쌍들이다.

### 신뢰도 숫자를 믿어도 되는가

믿어도 된다. 신뢰도 구간별로 실제 정확도를 갈라보면 거의 그대로 따라간다.

| 표시된 신뢰도 | 해당 이미지 | 실제 정확도 |
|---|---:|---:|
| 0 ~ 30% | 10장 | 50.0% |
| 30 ~ 60% | 12장 | 66.7% |
| 60 ~ 90% | 31장 | 80.6% |
| 90 ~ 99% | 52장 | 90.4% |
| 99% 이상 | 95장 | **98.9%** |

평균 신뢰도 88.2% vs 실제 정확도 89.5% — 과신하지도 과소평가하지도 않는다.
**화면의 퍼센트를 "이 정도로 맞을 확률"로 읽어도 무리가 없다.**

### 한계

- **1000개 클래스 밖은 분류할 수 없다.** ImageNet-1k 에는 사람 클래스가 없어서
  인물 사진을 넣으면 옷이나 배경 사물로 답한다. 한국 음식처럼 목록에 없는 대상도
  마찬가지로 가장 비슷한 다른 클래스를 고른다
- **표본이 200장이라 Top-1 의 표준오차가 ±2%p 수준**이다. 백엔드 간 0.5%p 차이는
  노이즈로 봐야 한다
- 평가에 쓴 이미지는 ImageNet 공식 validation set 이 아니라 클래스당 1장짜리
  샘플 저장소다. ViT-Base 의 공식 Top-1(81.1%, val 50,000장)과 직접 비교할 수 없다

---

## 화면

### 분류

위 스크린샷의 화면이다. 이미지 여러 장(최대 32장)을 한 번에 올려 배치 추론하고,
올린 것을 전부 썸네일로 미리 보여준다. 결과 카드마다 Top-1 라벨·신뢰도·지연시간과
**Top-K 막대 차트**를 내고, 전체 결과를 CSV / JSON 으로 내려받을 수 있다.
카메라 촬영 입력도 지원한다.

손상된 파일이 섞여 있어도 그 파일만 사유와 함께 건너뛰고 나머지는 계속 처리한다.

### 백엔드 비교

이미지 한 장을 선택한 백엔드들에 모두 통과시켜 **예측이 서로 일치하는지**와
**기준선 대비 속도 배율**을 나란히 놓는다. 아래에서 PyTorch INT8 이 `0.46× 속도`
로 표시된 것에 주목 — 양자화했는데 느려진 경우다. 요약 문장은 화면에 뜬 측정값에서
계산한다.

![백엔드 비교 화면](assets/screenshots/02-compare.png)

> 같은 고양이 사진을 네 백엔드에 통과시킨 결과. 넷 다 `tabby cat` 으로 같은 답을
> 냈지만 소요 시간은 36.0ms 에서 105.1ms 까지 벌어진다.

### 벤치마크

`results/benchmark.json` 을 표와 차트로 렌더링한다. 측정 환경(OS·Python·스레드 수)을
함께 적어둔다 — 다른 CPU 에서 잰 수치는 비교 의미가 없기 때문이다.

![벤치마크 화면](assets/screenshots/03-benchmark.png)

---

## 같은 모델, 네 가지 실행 방법

여기부터는 분류 정확도가 아니라 **그 추론을 어떻게 실행할 것인가**에 대한 이야기다.
모델과 예측 결과는 위와 동일하고, 달라지는 것은 런타임·정밀도와 그에 따른
속도·용량이다.

실제 배포 서버(OCI Ampere A1, Neoverse-N1 4 OCPU), 4스레드 고정,
ImageNet-1k 에서 균등 추출한 200장 기준.

| 변형 | 용량 | 용량 절감 | 지연 mean | 지연 p95 | 속도 | Top-1 | Top-5 | FP32 일치 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PyTorch FP32 | 330.3 MB | 0.0% | 272.9 ms | 409.7 ms | 1.00x | 88.0% | 99.0% | 100.0% |
| PyTorch INT8 (dynamic) | 85.2 MB | 74.2% | 467.0 ms | 476.0 ms | **0.58x** | 87.5% | 99.0% | 96.5% |
| ONNX FP32 | 330.4 MB | -0.0% | 273.6 ms | 319.4 ms | 1.00x | 88.0% | 99.0% | 100.0% |
| **ONNX INT8 (dynamic)** | **83.7 MB** | **74.7%** | **96.0 ms** | 97.3 ms | **2.84x** | 88.5% | 98.0% | 96.5% |

재현: `python scripts/run_benchmark.py --n 200 --runs 30 --threads 4`
원본 데이터: [`results/benchmark.json`](results/benchmark.json) (배포 서버) ·
[`results/benchmark-macos-arm.json`](results/benchmark-macos-arm.json) (개발 머신)

### 여기서 읽어야 할 것

**1. 양자화는 항상 빠르지 않다 — 하드웨어가 결론을 뒤집는다.**

세 플랫폼에서 같은 코드로 측정한 지연시간 배율이다.

| 지연시간 배율 (각 환경의 FP32=1.00) | Windows x86<br>fbgemm | macOS ARM (M5)<br>qnnpack | Linux ARM (Neoverse-N1)<br>qnnpack |
|---|---:|---:|---:|
| PyTorch INT8 | 1.65x | **0.43x** | **0.58x** |
| ONNX INT8 | 1.81x | 1.18x | **2.84x** |

PyTorch 동적 INT8 은 **ARM 두 곳 모두에서 FP32 보다 느렸다.** 용량은 74% 줄었는데
속도는 손해다. ARM 의 양자화 백엔드인 `qnnpack` 은 ViT 처럼 큰 Linear 가 연속되는
구조에서 양자화·역양자화 오버헤드를 FP32 경로로 상쇄하지 못한다. 반면 x86(`fbgemm`)
에서는 1.65배 빨랐다 ([vit-onnx-optimization](https://github.com/carsy078-maker/vit-onnx-optimization)).
"INT8 양자화 = 가속" 이라는 통념이 플랫폼에 따라 그대로 뒤집힌다.

**2. 최적화 효과는 배포 환경에서 재야 한다.**

ONNX INT8 의 이득이 개발 머신(M5)에서는 1.18배였는데 **실제 배포 하드웨어에서는
2.84배**였다. M5 는 FP32 경로가 워낙 빨라 INT8 로 얻을 여지가 적었던 반면,
서버급 Neoverse-N1 에서는 격차가 크게 벌어진다. 개발 머신 수치만 보고
"별 차이 없네" 라고 판단했다면 실제 서비스에서 **272.9ms → 96.0ms** 의 개선을
놓쳤을 것이다.

**3. ONNX Runtime 은 세 플랫폼 모두에서 안전한 선택이었다.**

ONNX INT8 은 어디서도 기준선보다 느려지지 않으면서 용량을 74.7% 줄였고,
FP32 와 Top-1 예측이 96.5% 일치했다. 그래서 이 프로젝트의 **배포 기본 백엔드**다.

**4. 양자화가 분류 품질을 해치지는 않았다.**

표의 Top-1 이 88.0% 와 88.5% 로 갈리지만 [앞서 적은 대로](#한계) 표본 오차 범위라
"INT8 이 더 정확하다"고 읽으면 안 된다. 의미 있는 수치는 **FP32 일치율 96.5%** 다 —
200장 중 193장에서 FP32 와 완전히 같은 Top-1 을 냈고, 갈린 7장도 대부분
`Appenzeller`↔`Bernese mountain dog` 처럼 FP32 도 헷갈리던 경계 사례다.
**용량을 74.7% 줄이고 2.84배 빨라지면서 치른 대가가 그 정도**라는 뜻이다.

---

## 설계에서 신경 쓴 것

### 환경별 백엔드 폴백

FP32 가중치는 한 종에 330MB 라 배포 이미지에 넣을 수 없다. 그래서 가중치
확보 경로를 세 갈래로 두고, 쓸 수 있는 백엔드를 런타임에 판정한다.

| 경로 | 설명 |
|---|---|
| 로컬 `models/` | `scripts/export_models.py` 로 내보낸 파일 |
| Hugging Face Hub | 배포 환경에서 ONNX INT8(84MB)만 내려받는다 |
| 원본에서 즉석 구성 | PyTorch 변형은 파일 없이 `from_pretrained` 로 만든다 |

쓸 수 없는 변형은 숨기지 않고 **사유와 함께** UI에 남긴다
(`"이 환경에는 torch 가 없다 (ONNX 전용 배포)"`). 판정 로직은
[`backends.variant_status()`](src/vision_dashboard/backends.py) 한곳에 모여 있고,
UI 는 결과를 읽기만 한다.

### 배포 의존성에서 torch 를 뺐다

전처리를 transformers 의 PIL 백엔드로 고정해, 배포 환경은 **ONNX Runtime 만으로**
동작한다. `requirements.txt` 에 torch 가 없다.

torchvision 백엔드와 PIL 백엔드는 리사이즈 구현이 미묘하게 다르다(같은 이미지에서
최대 0.008 차이). 백엔드를 고정하지 않으면 개발 환경과 배포 환경이 서로 다른 입력을
받게 되고, 위의 벤치마크 수치가 배포 환경에서 재현되지 않는다.
[CI 가 이 조합을 매번 검증한다](.github/workflows/ci.yml) — 배포 의존성에 torch 가
섞이면 빌드가 깨진다.

### 추론 결과 캐싱

Streamlit 은 위젯이 바뀔 때마다 스크립트를 처음부터 다시 실행한다. 확률 1000개를
통째로 들고 있으므로 **Top-K 개수를 바꾸거나 이모지를 켜고 꺼도 재추론하지 않는다.**
재추론은 새 이미지가 들어오거나 백엔드를 바꿀 때만 일어난다.

### 공정한 측정

PyTorch 와 ONNX Runtime 의 스레드 수를 같은 값으로 강제한다. 기본값을 두면 각자
다른 수의 코어를 잡아, 런타임 비교가 아니라 스레드 수 비교가 되어 버린다.
지연시간은 warmup 후 30회 반복해 mean / p50 / p95 를 낸다.

### 평가 표본 추출

ImageNet 클래스 인덱스는 의미 순으로 정렬돼 있다(0~397 이 대부분 동물, 그중 앞쪽은
어류·조류). 앞에서 200개를 자르면 사실상 "동물 분류기" 를 평가하게 되므로,
전체 1000 클래스에서 **균등 간격으로** 추출한다.

---

## 구조

```
app.py                       Streamlit 엔트리
src/vision_dashboard/
├── config.py                모델 변형 정의 · 실행 환경 프로파일
├── backends.py              런타임 추상화 (PyTorch/ONNX × FP32/INT8) · 가용성 판정
├── inference.py             이미지 로드 · 전처리 · 배치 추론 · 후처리
├── labels.py                ImageNet-1k 라벨 · 이모지 매핑
├── benchmarking.py          용량/지연/정확도 측정
└── ui/                      Streamlit 화면 (분류 · 백엔드 비교 · 벤치마크)
scripts/
├── export_models.py         4종 가중치 내보내기
├── download_samples.py      평가용 이미지 다운로드
├── run_benchmark.py         벤치마크 실행
└── capture_screenshots.py   README 스크린샷 캡처 (Playwright)
tests/                       pytest (61개, Streamlit AppTest 포함)
```

---

## 실행

### 로컬 (4종 전부)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

python scripts/export_models.py          # 가중치 4종 생성 (~830MB)
streamlit run app.py
```

### 배포 구성 재현 (ONNX 전용, torch 없음)

```bash
pip install -r requirements.txt
python scripts/export_models.py --only onnx_int8   # 별도 환경에서 미리 생성 필요
streamlit run app.py
```

Hub 에 ONNX 가중치를 올려두고 내려받게 하려면:

```bash
export VD_HUB_ONNX_REPO="<사용자명>/vit-base-onnx-int8"
```

### Docker

```bash
docker compose up -d --build                  # 앱만 (127.0.0.1:8501 바인딩)
docker compose --profile standalone up -d     # 앞단 Caddy 까지 함께
```

기본 구성은 앱을 루프백에만 연다. 호스트에 이미 웹 서버가 있는 서버에서 컨테이너가
80 을 뺏지 않도록 한 것으로, 그런 환경에서는 기존 프록시에 한 블록만 더하면 된다.

```caddy
vit.example.com {
    reverse_proxy 127.0.0.1:8501
}
```

[Dockerfile](Dockerfile) 은 다단계 빌드이고 최종 타깃이 둘이다. builder 단계에서만
torch 를 설치해 가중치를 내보내고, 런타임 이미지는 목적에 따라 갈린다.

| 타깃 | 담는 것 | 쓸 수 있는 백엔드 | 용도 |
|---|---|---|---|
| `runtime-onnx` | ONNX INT8(84MB) + ONNX Runtime | 2종 중 1종 | 메모리가 빠듯한 PaaS |
| `runtime-full` | 4종 전부 + torch | **4종 전부** | 여유 있는 서버 (compose 기본값) |

```bash
docker build --target runtime-onnx -t vision-dashboard:onnx .   # 경량
```

어느 쪽이든 앱 코드는 같다. 쓸 수 있는 백엔드가 환경마다 달라지는 것을
`backends.variant_status()` 가 판정하고, 못 쓰는 변형은 사유와 함께 화면에 남는다.

### 벤치마크

```bash
python scripts/download_samples.py --n 200
python scripts/run_benchmark.py --n 200 --runs 30 --threads 4
```

### 테스트

```bash
pytest -q          # 61 passed
ruff check .
```

모델 가중치 없이 도는 테스트다. 백엔드는 목으로 대체하고, 배치 분할·순서 보존·
예외 처리·가용성 판정 같은 배선을 검증한다.

UI 는 Streamlit `AppTest` 로 검증한다. 그중 하나는 **Top-K 슬라이더와 이모지
토글을 움직여도 추론 함수가 다시 호출되지 않는지** 를 호출 횟수로 확인한다 —
원본 앱에서 실제로 깨져 있던 동작이라 회귀 방지용으로 고정해 뒀다.

---

## 환경 변수

| 이름 | 기본값 | 설명 |
|---|---|---|
| `VD_HUB_ONNX_REPO` | (없음) | ONNX INT8 가중치를 받을 Hugging Face 저장소 |
| `VD_CPU_THREADS` | `min(4, 코어 수)` | PyTorch/ONNX 공통 스레드 수 |

---

## 기술 스택

Python 3.12+ · Streamlit · ONNX Runtime · PyTorch · Hugging Face Transformers ·
Altair · pytest · GitHub Actions · Docker

---

## 라이선스

MIT
