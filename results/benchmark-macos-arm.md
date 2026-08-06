# 벤치마크 결과

- 모델: `google/vit-base-patch16-224`
- 평가 이미지: 200장 (ImageNet-1k 에서 균등 추출)
- 환경: macOS-26.5.1-arm64-arm-64bit, Python 3.12.13, CPU 4 threads (PyTorch/ONNX 동일)

| 변형 | 용량 | 용량 절감 | 지연 mean | 지연 p95 | 속도 | Top-1 | Top-5 | FP32 일치 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PyTorch FP32 | 330.3 MB | 0.0% | 39.2 ms | 39.9 ms | 1.00x | 88.0% | 99.0% | 100.0% |
| PyTorch INT8 (dynamic) | 85.2 MB | 74.2% | 90.2 ms | 93.0 ms | 0.43x | 88.0% | 98.5% | 96.0% |
| ONNX FP32 | 330.4 MB | -0.0% | 35.6 ms | 39.9 ms | 1.10x | 88.0% | 99.0% | 100.0% |
| ONNX INT8 (dynamic) | 83.7 MB | 74.7% | 33.2 ms | 33.7 ms | 1.18x | 89.5% | 99.0% | 97.0% |
