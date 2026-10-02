"""Check date alignment, missing observations and actual model configuration."""
import unittest
from unittest.mock import patch

import pandas as pd

from Appendix_Fig_main import current_hyperparameter_table, descriptive_table
from experiment_core import DATA_DIR, TEST_END


class AppendixTablesTest(unittest.TestCase):
    def test_b4_excludes_trading_days_after_march_3(self):
        frame = pd.DataFrame({
            "trade_date": ["2023-03-03", "2023-03-06", "2023-03-07", "2023-03-08"],
            "open": [1, 2, 3, 1000],
        })
        regular = descriptive_table(frame, "trade_date", ["open"])
        self.assertEqual(TEST_END, 20230303)
        self.assertEqual(regular.iloc[0]["Number of trading days"], 1)
        self.assertEqual(regular.iloc[0]["Max"], "2023/3/3")
        self.assertEqual(regular.iloc[1]["Mean"], 1)

    def test_b4_uses_original_1272_trading_days(self):
        base = pd.read_csv(DATA_DIR / "dapan" / "3068merge.csv")
        table = descriptive_table(base, "trade_date", ["open", "group_len"])
        self.assertEqual(table.iloc[0]["Max"], "2023/3/3")
        self.assertTrue((table["Number of trading days"] == 1272).all())

    def test_trading_date_includes_first_day_despite_previous_report_date(self):
        frame = pd.DataFrame({
            "trade_date": ["2017-12-05", "2017-12-06", "2023-03-03", "2023-03-04"],
            "qid_date": [20171204, 20171205, 20230302, 20230303],
            "open": [1000, 1, 3, 1000],
        })
        table = descriptive_table(frame, "trade_date", ["open"])
        self.assertEqual(table.iloc[0]["Min"], "2017/12/6")
        self.assertEqual(table.iloc[0]["Max"], "2023/3/3")
        self.assertEqual(table.iloc[1]["Mean"], 2)
        self.assertAlmostEqual(table.iloc[1]["Std."], 2 ** 0.5)
        self.assertEqual(table.iloc[1]["Number of trading days"], 2)

    def test_report_count_does_not_impute_missing_feature(self):
        frame = pd.DataFrame({"qid_date": [20171205, 20171206, 20171207, 20230303],
                              "page": [999, 1, None, 3]})
        table = descriptive_table(frame, "qid_date", ["page"])
        self.assertEqual(table.iloc[0]["Number of samples"], 3)
        self.assertEqual(table.iloc[1]["Number of samples"], 2)
        self.assertEqual(table.iloc[1]["Mean"], 2)

    def test_selected_rank_rate_tracks_configuration_not_manuscript_literal(self):
        with patch.dict("Appendix_Fig_main.PAPER_HYPERPARAMETERS",
                        {"LambdaRank": {"0060": {"learning_rate": 0.037},
                                        "3068": {"learning_rate": 0.019}}}):
            table = current_hyperparameter_table()
        rates = table[table.Models.str.startswith("LambdaRank")]["Selected choice"].tolist()
        self.assertEqual(rates, [0.037, 0.019])


if __name__ == "__main__":
    unittest.main()
