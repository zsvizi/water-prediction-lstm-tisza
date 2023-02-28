from __future__ import annotations

import pandas as pd

from src.evaluation.evaluation_base import EvaluationBase


class Evaluator(EvaluationBase):
    def __init__(self, validation_df: pd.DataFrame = None, prediction_df: pd.DataFrame = None):

        super().__init__()
        self.validation_df = validation_df
        self.prediction_df = prediction_df
        self.error_df = self.prediction_df - self.validation_df

    def calculate_all_stats(self, observed=None, modeled=None):
        if observed is None:
            observed = self.validation_df
        if modeled is None:
            modeled = self.prediction_df

        # calc_list = [rmse, mae, r2, nse, wi, mape, rmae, mse, rrmse, rmsre, nnse, anse, nanse, fc, lm, nwi, corr]

        stats = pd.DataFrame()
        stats["rmse"] = self.rmse(observed, modeled)
        stats["mae"] = self.mae(observed, modeled)
        stats["r2"] = self.r2(observed, modeled)
        stats["nse"] = self.nse(observed, modeled)
        stats["wi"] = self.wi(observed, modeled)
        stats["mape"] = self.mape(observed, modeled)
        stats["rmae"] = self.rmae(observed, modeled)
        stats["mse"] = self.mse(observed, modeled)
        stats["rrmse"] = self.rrmse(observed, modeled)
        stats["rmsre"] = self.rmsre(observed, modeled)
        stats["nnse"] = self.nnse(observed, modeled)
        stats["anse"] = self.anse(observed, modeled)
        stats["nanse"] = self.nanse(observed, modeled)
        stats["fc"] = self.fc(observed, modeled)
        stats["lm"] = self.lm(observed, modeled)
        stats["nwi"] = self.nwi(observed, modeled)
        stats["corr"] = self.corr(observed, modeled)
        return stats
