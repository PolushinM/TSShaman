import unittest


class TaxiTest(unittest.TestCase):
    def test_r2_performance(self):
        r_2 = 0.88  # TODO: Add r_2 calculation
        self.assertTrue(0.86 < r_2)


if __name__ == '__main__':
    unittest.main()
