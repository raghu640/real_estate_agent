import os
import pytest


@pytest.fixture(autouse=True, scope="session")
def stub_env_vars():
    os.environ.setdefault("OPENAI_API_KEY", "test-key-placeholder")
    os.environ.setdefault("APIFY_API_TOKEN", "test-token-placeholder")
