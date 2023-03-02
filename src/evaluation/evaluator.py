from __future__ import annotations

import pandas as pd

from src.evaluation.evaluation_base import EvaluationBase


class Evaluator(EvaluationBase):
    def __init__(self, validation_df: pd.DataFrame = None, prediction_df: pd.DataFrame = None):
        """
        This class calculates statistic.
        :param pd.DataFrame validation_df: the validation data
        :param pd.DataFrame prediction_df: the predicted data
        """

        super().__init__()
        self.validation_df = validation_df
        self.prediction_df = prediction_df
        self.error_df = self.prediction_df - self.validation_df

    def calculate_all_stats(self, observed: pd.Series = None, modeled: pd.Series = None) -> pd.DataFrame:
        """
        Calculates all methods, that are in the EvaluationBase class.
        :param pd.Series observed: The observed data. (validation)
        :param pd.Series modeled: The modeled data. (prediction)
        :return pd.DataFrame stats: the calculated statistic dataframe
        """
        if observed is None:
            observed = self.validation_df
        if modeled is None:
            modeled = self.prediction_df

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
