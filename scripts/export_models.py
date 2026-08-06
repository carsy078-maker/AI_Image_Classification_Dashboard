"""ViT-Base 를 4가지 배포 형식으로 내보낸다.

    python scripts/export_models.py
    python scripts/export_models.py --only onnx_int8 --force

생성물 (models/, git 추적 대상 아님):
    vit_fp32.pth            PyTorch state_dict (FP32 기준선)
    vit_int8_dynamic.pth    Linear 레이어 동적 INT8 양자화
    vit_fp32.onnx           ONNX 그래프 (동적 배치 축)
    vit_int8_dynamic.onnx   ONNX Runtime 동적 INT8 양자화
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from onnxruntime.quantization import QuantType, quantize_dynamic  # noqa: E402
from transformers import ViTForImageClassification  # noqa: E402

from vision_dashboard.backends import setup_quantization_engine  # noqa: E402
from vision_dashboard.config import (  # noqa: E402
    MODEL_NAME,
    MODELS_DIR,
    OPSET_VERSION,
    VARIANTS,
    ensure_dirs,
)
from vision_dashboard.labels import fetch_and_cache_labels  # noqa: E402


def human_size(path: Path) -> str:
    return f"{path.stat().st_size / (1024 * 1024):.2f} MB"


def load_fp32_model() -> ViTForImageClassification:
    model = ViTForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return model


def quantize_torch_dynamic(model: ViTForImageClassification) -> torch.nn.Module:
    """Linear 레이어만 INT8 로 동적 양자화한다.

    Attention/MLP 의 Linear 가 ViT 파라미터의 대부분을 차지하므로
    LayerNorm·임베딩을 FP32 로 남겨도 용량 감소 효과가 크다.
    """
    setup_quantization_engine()
    return torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def export_onnx(model: ViTForImageClassification, path: Path) -> None:
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["pixel_values"],
        output_names=["logits"],
        # 배치 축을 동적으로 열어둬야 서빙 시 배치 추론이 가능하다.
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        dynamo=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ViT 모델 4종 내보내기")
    parser.add_argument("--force", action="store_true", help="이미 있는 파일도 다시 생성")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(VARIANTS),
        help="지정한 변형만 생성 (기본: 전부)",
    )
    args = parser.parse_args()

    ensure_dirs()
    wanted = set(args.only or VARIANTS)
    targets = {k: VARIANTS[k]["path"] for k in wanted}

    if all(p.exists() for p in targets.values()) and not args.force:
        print("요청한 파일이 모두 존재한다. 다시 만들려면 --force 를 붙일 것.")
        for p in targets.values():
            print(f"  {p.name:<24} {human_size(p)}")
        return

    print("ImageNet 라벨 캐시 확인...")
    fetch_and_cache_labels()

    print(f"{MODEL_NAME} 로드 중...")
    t0 = time.perf_counter()
    model = load_fp32_model()
    print(f"  완료 ({time.perf_counter() - t0:.1f}s)")

    fp32_pth = VARIANTS["pytorch_fp32"]["path"]
    int8_pth = VARIANTS["pytorch_int8"]["path"]
    fp32_onnx = VARIANTS["onnx_fp32"]["path"]
    int8_onnx = VARIANTS["onnx_int8"]["path"]

    if "pytorch_fp32" in wanted and (args.force or not fp32_pth.exists()):
        print("[pytorch_fp32] state_dict 저장...")
        torch.save(model.state_dict(), fp32_pth)

    if "pytorch_int8" in wanted and (args.force or not int8_pth.exists()):
        print("[pytorch_int8] 동적 INT8 양자화...")
        # state_dict 만 저장한다. 로드할 때는 FP32 모델에 같은 양자화를 적용해
        # 동일한 구조를 만든 뒤 load_state_dict 한다 (backends.TorchBackend 참고).
        torch.save(quantize_torch_dynamic(model).state_dict(), int8_pth)

    # ONNX INT8 은 FP32 그래프에서 파생되므로 둘 중 하나만 요청해도 FP32 가 필요하다.
    need_fp32_onnx = "onnx_fp32" in wanted or "onnx_int8" in wanted
    if need_fp32_onnx and (args.force or not fp32_onnx.exists()):
        print("[onnx_fp32] ONNX 내보내기...")
        export_onnx(model, fp32_onnx)

    if "onnx_int8" in wanted and (args.force or not int8_onnx.exists()):
        print("[onnx_int8] ONNX Runtime 동적 INT8 양자화...")
        quantize_dynamic(
            model_input=str(fp32_onnx),
            model_output=str(int8_onnx),
            weight_type=QuantType.QInt8,
        )

    print(f"\n생성 완료 -> {MODELS_DIR}")
    for key in VARIANTS:
        p = VARIANTS[key]["path"]
        if p.exists():
            print(f"  {p.name:<24} {human_size(p)}")


if __name__ == "__main__":
    main()
