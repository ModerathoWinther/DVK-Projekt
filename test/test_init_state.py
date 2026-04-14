import unittest
from src import init_state as init

PARAMS = ['train', True, True, True]

class TestInitEnvironment(unittest.TestCase):

    def test_market_data_has_correct_dimensions(self):
        market_data = init.get_market_data(*PARAMS)

        expected_width = 10
        actual_height, actual_width = market_data.shape

        self.assertEqual(actual_width, expected_width,
                         f"Expected {expected_width} columns, but got {actual_width}")
        self.assertGreater(actual_height, 0, "Market data is empty!")
