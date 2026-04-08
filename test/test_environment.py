import pytest

from src import environment

@pytest.fixture(scope="session")
def training_from_date():
    return "2016-01-01"

@pytest.fixture(scope="session")
def training_to_date():
    return "2022-12-31"

def test_reward_calculation():
    assert environment.MARKET_HIGH == 1