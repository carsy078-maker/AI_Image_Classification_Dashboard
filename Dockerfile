# 다단계 빌드. 최종 타깃이 둘이다.
#
#   runtime-onnx  ONNX INT8(84MB) 하나만 싣는 경량 이미지. torch 가 없다.
#                 메모리가 빠듯한 PaaS 를 상정한 구성.
#   runtime-full  4종 전부. 백엔드 비교 화면이 온전히 동작한다.
#                 여유 있는 서버(예: OCI Ampere A1 4 OCPU / 24GB)용 기본값.
#
#   docker build --target runtime-onnx -t vision-dashboard:onnx .
#   docker build --target runtime-full -t vision-dashboard:full .
#
# 어느 쪽이든 앱 코드는 같다. 쓸 수 있는 백엔드가 환경에 따라 달라지는 것을
# 코어(backends.variant_status)가 알아서 판정하고 UI 에 사유까지 표시한다.

ARG PYTHON_VERSION=3.12

# ---------- builder: 모델 내보내기 전용 ----------
FROM python:${PYTHON_VERSION}-slim AS builder

WORKDIR /build

# CPU 전용 torch 휠. 기본 인덱스의 x86 torch 는 CUDA 런타임을 포함해 수 GB 다.
# (aarch64 는 애초에 CPU 휠만 배포되므로 이 인덱스는 무시된다.)
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.13.0 transformers==5.14.1 onnx==1.22.0 onnxruntime==1.28.0 \
    numpy==2.5.1 pillow==12.3.0 huggingface-hub==1.26.0

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY assets/imagenet_classes.json ./assets/imagenet_classes.json

# 4종 전부 만든다. 경량 타깃은 이 중 ONNX INT8 하나만 가져간다.
RUN python scripts/export_models.py

# ---------- 공통 런타임 베이스 ----------
FROM python:${PYTHON_VERSION}-slim AS runtime-base

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    VD_CPU_THREADS=4

COPY src/ ./src/
COPY assets/ ./assets/
COPY results/ ./results/
COPY .streamlit/ ./.streamlit/
COPY app.py .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# ---------- runtime-onnx: 경량 (ONNX INT8 only) ----------
FROM runtime-base AS runtime-onnx

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=builder /build/models/vit_int8_dynamic.onnx ./models/

# ---------- runtime-full: 4종 전부 ----------
FROM runtime-base AS runtime-full

# torch 가 있어야 PyTorch 변형 2종을 쓸 수 있다. ONNX 전용 구성과 달리
# 이미지가 무거워지는 대신 백엔드 비교가 4종 전부로 성립한다.
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt torch==2.13.0

COPY --from=builder /build/models/ ./models/
