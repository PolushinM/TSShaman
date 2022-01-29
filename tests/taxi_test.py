import unittest
import importlib.util
import pandas as pd
from sklearn.metrics import r2_score
import warnings
import sys
from pathlib import Path

# import TSShaman as sh
sys.path.append(str(Path(__file__).parent.parent))
import TSShaman as sh


class TaxiTest(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore")  # TODO: fix warnings
        data = pd.read_csv('ESB_cell.csv', parse_dates=[0], index_col=0)
        self.X = data.drop('1231', axis=1)
        self.y = data['1231']
        return

    def test_r2_performance(self):
        model = sh.TSShaman(review_period=505, forecast_horizon=250, verbosity='error')
        predicted_y = model.fit(self.X[:-300],
                                self.y[:-300],
                                cv=2,
                                omega=0.13
                                ).predict(forecast_period=300)

        self.assertTrue(r2_score(self.y[-300:], predicted_y) > 0.86)


if __name__ == '__main__':
    unittest.main()
