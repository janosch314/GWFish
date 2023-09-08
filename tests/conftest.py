import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--plot", action="store_true", default=False, help="produce plots for the tests that support them"
    )


@pytest.fixture
def plot(request):
    return request.config.getoption("--plot")
