"""README 용 스크린샷을 자동으로 찍는다.

    streamlit run app.py --server.port 8599 --server.headless true &
    python scripts/capture_screenshots.py

준비물: `pip install playwright && playwright install chromium`
그리고 `scripts/export_models.py` 로 만든 가중치 4종 (백엔드 비교 화면에 필요).

Streamlit 은 body 가 아니라 내부 컨테이너가 스크롤되므로 full_page 옵션만으로는
뷰포트 밖이 잘린다. 콘텐츠 높이를 재서 뷰포트를 그만큼 키운 뒤 찍는다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "screenshots"

WIDTH = 1500
FINAL_WIDTH = 1600  # README 에 넣을 최종 가로폭
MAX_HEIGHT = 7000

# 분류 화면에 쓸 이미지. 예측 라벨이 서로 겹치지 않게 고른다.
CLASSIFY_IMAGES = [
    "data/samples/0205_n02099267.jpg",  # flat-coated retriever
    "data/samples/0340_n02391049.jpg",  # zebra
    "data/samples/0550_n03297495.jpg",  # espresso maker
    "data/samples/0985_n11939491.jpg",  # daisy
]
COMPARE_IMAGE = "tests/fixtures/test_image.jpg"  # tabby cat

CONTENT_HEIGHT = """() => {
    const main = document.querySelector('[data-testid="stMain"]');
    const inner = document.querySelector('[data-testid="stMainBlockContainer"]');
    return Math.max(
        main ? main.scrollHeight : 0,
        inner ? inner.getBoundingClientRect().height + 120 : 0,
        document.body.scrollHeight,
    );
}"""


def settle(page, ms: int = 2500) -> None:
    """Streamlit 의 렌더 사이클이 끝나기를 기다린다.

    실행 중 표시가 사라질 때까지 본 뒤 여유를 준다. 차트(Vega)는 DOM 이 붙은
    뒤에도 그려지는 데 시간이 걸린다.
    """
    try:
        page.wait_for_selector('[data-testid="stStatusWidget"]', state="detached", timeout=300_000)
    except Exception:
        pass
    page.wait_for_timeout(ms)


def shoot(page, name: str) -> None:
    # 버튼 위에 커서가 남아 있으면 툴팁이 떠서 화면을 가린다.
    page.mouse.move(4, 4)
    page.keyboard.press("Escape")
    page.wait_for_timeout(700)

    height = int(page.evaluate(CONTENT_HEIGHT))
    page.set_viewport_size({"width": WIDTH, "height": min(height + 60, MAX_HEIGHT)})
    page.wait_for_timeout(1500)  # 뷰포트가 바뀌면 차트가 다시 그려진다

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    page.screenshot(path=str(path), full_page=True)

    # 레티나 배율로 찍은 뒤 줄여야 글자가 또렷하다. 그대로 두면 3000px 짜리가 된다.
    with Image.open(path) as img:
        if img.width > FINAL_WIDTH:
            img = img.resize(
                (FINAL_WIDTH, int(img.height * FINAL_WIDTH / img.width)), Image.LANCZOS
            )
        img.convert("RGB").save(path, "PNG", optimize=True)

    with Image.open(path) as img:
        size = img.size
    print(f"  저장 {path.relative_to(ROOT)}  {size[0]}x{size[1]}  {path.stat().st_size // 1024} KB")

    page.set_viewport_size({"width": WIDTH, "height": 1000})
    page.wait_for_timeout(800)


def main() -> None:
    parser = argparse.ArgumentParser(description="README 스크린샷 캡처")
    parser.add_argument("--url", default="http://localhost:8599", help="실행 중인 앱 주소")
    args = parser.parse_args()

    missing = [p for p in [*CLASSIFY_IMAGES, COMPARE_IMAGE] if not (ROOT / p).exists()]
    if missing:
        raise SystemExit(
            "필요한 이미지가 없다: "
            + ", ".join(missing)
            + "\n`python scripts/download_samples.py --n 200` 을 먼저 실행할 것."
        )

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(
            viewport={"width": WIDTH, "height": 1000},
            device_scale_factor=2,
        )
        page.goto(args.url, wait_until="networkidle", timeout=120_000)
        settle(page, 3000)

        print("[1/3] 분류 탭")
        page.locator('input[type="file"]').first.set_input_files(
            [str(ROOT / p) for p in CLASSIFY_IMAGES]
        )
        settle(page, 2500)
        page.get_by_role("button", name="분석 실행").click()
        settle(page, 4000)
        shoot(page, "01-classify.png")

        print("[2/3] 백엔드 비교 탭")
        page.get_by_role("tab", name="백엔드 비교").click()
        settle(page, 2000)
        page.locator('input[type="file"]').last.set_input_files(str(ROOT / COMPARE_IMAGE))
        settle(page, 2500)
        page.get_by_role("button", name="비교 실행").click()
        settle(page, 5000)  # PyTorch FP32 최초 로드가 수십 초 걸린다
        shoot(page, "02-compare.png")

        print("[3/3] 벤치마크 탭")
        page.get_by_role("tab", name="벤치마크").click()
        settle(page, 4000)
        shoot(page, "03-benchmark.png")

        browser.close()


if __name__ == "__main__":
    sys.exit(main())
