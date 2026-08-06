from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_image_path() -> Path:
    return FIXTURES / "test_image.jpg"


@pytest.fixture(scope="session")
def sample_image_bytes(sample_image_path: Path) -> bytes:
    return sample_image_path.read_bytes()
