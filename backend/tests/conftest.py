import os
import tempfile
from pathlib import Path

import pytest

# Point the app at throwaway dirs before it is imported. MODELS_DIR is empty on
# purpose: tests must pass on machines without model weights.
_tmp = Path(tempfile.mkdtemp(prefix="annotator-test-"))
os.environ["DATA_DIR"] = str(_tmp / "data")
os.environ["MODELS_DIR"] = str(_tmp / "models")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c
