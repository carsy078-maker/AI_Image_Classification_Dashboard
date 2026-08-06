"""ImageNet-1k 평가용 샘플 이미지를 내려받는다.

    python scripts/download_samples.py --n 200

EliSchwartz/imagenet-sample-images 저장소는 1000개 클래스마다 이미지 1장씩을
`n01440764_tench.JPEG` 형식으로 담고 있다. 파일명 앞의 WordNet ID 를 오름차순
정렬한 순서가 곧 ImageNet-1k 표준 클래스 인덱스이므로, 별도 라벨 파일 없이도
정답 클래스를 얻을 수 있다.

전체 1000개 중 일부만 받을 때는 앞에서부터 자르지 않고 균등 간격으로 뽑는다.
ImageNet 클래스 인덱스는 의미 순으로 정렬돼 있어서(0~397 이 대부분 동물,
그중 앞쪽은 어류·조류) 앞에서 200개를 자르면 '동물 200종 분류기' 를 평가하는
셈이 된다. 그렇게 재면 정확도가 실제보다 높게 나온다.

내려받은 이미지는 data/samples/ 에 `{클래스인덱스:04d}_{wnid}.jpg` 로 저장하고,
목록은 data/samples/index.json 에 기록한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import requests  # noqa: E402

from vision_dashboard.config import SAMPLES_DIR, ensure_dirs  # noqa: E402

TREE_API = "https://api.github.com/repos/EliSchwartz/imagenet-sample-images/git/trees/master"
RAW_BASE = "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/"


def list_remote_files() -> list[str]:
    resp = requests.get(TREE_API, timeout=30)
    resp.raise_for_status()
    tree = resp.json()["tree"]
    files = [n["path"] for n in tree if n["path"].lower().endswith(".jpeg")]
    # WordNet ID 오름차순 = ImageNet-1k 표준 클래스 순서
    return sorted(files)


def pick_indices(total: int, n: int) -> list[int]:
    """0..total-1 에서 균등 간격으로 n 개를 고른다."""
    n = min(n, total)
    if n <= 0:
        return []
    step = total / n
    return sorted({min(total - 1, int(i * step)) for i in range(n)})


def download_one(job: tuple[int, str]) -> dict | None:
    class_idx, remote_name = job
    wnid = remote_name.split("_", 1)[0]
    dest = SAMPLES_DIR / f"{class_idx:04d}_{wnid}.jpg"
    record = {"path": dest.name, "label": class_idx, "wnid": wnid}
    if dest.exists() and dest.stat().st_size > 0:
        return record
    try:
        resp = requests.get(RAW_BASE + remote_name, timeout=60)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return record
    except Exception as exc:  # 개별 실패는 건너뛰고 나머지를 계속 받는다
        print(f"  건너뜀 {remote_name}: {exc}", file=sys.stderr)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="ImageNet 샘플 이미지 다운로드")
    parser.add_argument("--n", type=int, default=200, help="받을 이미지 수 (최대 1000)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    ensure_dirs()

    print("파일 목록 조회 중...")
    files = list_remote_files()
    indices = pick_indices(len(files), args.n)
    # 클래스 인덱스는 항상 전체 정렬 기준으로 매긴다.
    jobs = [(i, files[i]) for i in indices]

    print(f"{len(jobs)}개 이미지 다운로드 (전체 {len(files)}클래스에서 균등 추출) -> {SAMPLES_DIR}")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = [r for r in pool.map(download_one, jobs) if r]

    index_path = SAMPLES_DIR / "index.json"
    index_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"완료: {len(records)}개, 목록 -> {index_path}")


if __name__ == "__main__":
    main()
