import unittest
from gait_classification.main import main


class MainTest(unittest.TestCase):
    def test_main_runs(self):
        result = main()
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
