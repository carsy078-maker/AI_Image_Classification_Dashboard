# 벤치마크 결과

- 모델: `google/vit-base-patch16-224`
- 평가 이미지: 200장 (ImageNet-1k 에서 균등 추출)
- 환경: Linux-6.17.0-1011-oracle-aarch64-with-glibc2.41, Python 3.12.13, CPU 4 threads (PyTorch/ONNX 동일)

| 변형 | 용량 | 용량 절감 | 지연 mean | 지연 p95 | 속도 | Top-1 | Top-5 | FP32 일치 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PyTorch FP32 | 330.3 MB | 0.0% | 272.9 ms | 409.7 ms | 1.00x | 88.0% | 99.0% | 100.0% |
| PyTorch INT8 (dynamic) | 85.2 MB | 74.2% | 467.0 ms | 476.0 ms | 0.58x | 87.5% | 99.0% | 96.5% |
| ONNX FP32 | 330.4 MB | -0.0% | 273.6 ms | 319.4 ms | 1.00x | 88.0% | 99.0% | 100.0% |
| ONNX INT8 (dynamic) | 83.7 MB | 74.7% | 96.0 ms | 97.3 ms | 2.84x | 88.5% | 98.0% | 96.5% |
