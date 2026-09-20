import pytest
import torch

from fluxwar.config import load


def pytest_configure(config):
    torch.manual_seed(0)
    config.addinivalue_line("markers", "slow: needs the real LW6 kernel and takes seconds")


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def cfg():
    return load()
