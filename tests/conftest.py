import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import build_assistant, create_app
from scripts.make_sample_pdf import make


@pytest.fixture
def settings(tmp_path):
    s = Settings(
        _env_file=None,
        llm_provider="fake",
        embedding_provider="hash",
        data_dir=tmp_path,
        admin_api_key="test-key",
        min_similarity=0.2,
    )
    s.uploads_dir.mkdir(parents=True, exist_ok=True)
    return s


@pytest.fixture
def sample_pdf(tmp_path):
    return make(tmp_path / "sample.pdf")


@pytest.fixture
def assistant(settings):
    return build_assistant(settings)


@pytest.fixture
def indexed(assistant, sample_pdf):
    assistant.db.add_document("doc1", "Sample SOP", "1.0", str(sample_pdf))
    assistant.index_document("doc1", "Sample SOP", "1.0", sample_pdf)
    return assistant


@pytest.fixture
def client(settings, assistant):
    return TestClient(create_app(settings, assistant))
