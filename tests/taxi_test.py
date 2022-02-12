import unittest
import warnings
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import r2_score

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
        model = sh.TSShaman(review_period=400, forecast_horizon=300, verbosity='debug')
        predicted_y = model.fit(self.X[:-300],
                                self.y[:-300],
                                cv=2,
                                omega=0.5
                                ).predict(forecast_segment=300)
        print(f'Test R^2={r2_score(self.y[-300:], predicted_y):.5f}\n\n')
        self.assertGreater(r2_score(self.y[-300:], predicted_y), 0.75)

        model = sh.TSShaman(review_period=673, forecast_horizon=300, verbosity='debug', random_state=0)
        predicted_y = model.fit(self.X[:-300],
                                self.y[:-300],
                                cv=2,
                                omega=0.2
                                ).predict(forecast_segment=300)

        print(f'Test R^2={r2_score(self.y[-300:], predicted_y):.5f}')
        self.assertGreater(r2_score(self.y[-300:], predicted_y), 0.85)


if __name__ == '__main__':
    unittest.main()
