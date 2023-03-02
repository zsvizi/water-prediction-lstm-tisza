import numpy as np
import pandas as pd


class EvaluationBase:
    def __init__(self):
        pass

    # Nash-Sutcliffe efficiency NSE
    @staticmethod
    def nse(observed: pd.Series, modeled: pd.Series) -> pd.Series:
        """
        Calculate the Nash-Sutcliffe efficiency.
        :param pd.Series observed: The observed data. (validation)
        :param pd.Series modeled: The modeled data. (prediction)
        :return pd.Series: the calculated Nash-Sutcliffe efficiency
        """
        nse = 1 - (np.sum((observed - modeled) ** 2) / np.sum((observed - np.mean(observed)) ** 2))
        return nse

    @staticmethod
    def nnse(observed: pd.Series, modeled: pd.Series) -> pd.Series:
        """
        Calculate the normalized Nash-Sutcliffe efficiency.
        :param pd.Series observed: The observed data. (validation)
        :param pd.Series modeled: The modeled data. (prediction)
        :return pd.Series: the calculated normalized Nash-Sutcliffe efficiency
        """
        nse = 1 - (np.sum((observed - modeled) ** 2) / np.sum((observed - np.mean(observed)) ** 2))
        return 1 / (2 - nse)

    @staticmethod
    def anse(observed: pd.Series, modeled: pd.Series) -> pd.Series:
        """
        Calculate a modified version of Nash-Sutcliffe efficiency with absolute values to handle outliers.
        :param pd.Series observed: The observed data. (validation)
        :param pd.Series modeled: The modeled data. (prediction)
        :return pd.Series: the calculated modified version of Nash-Sutcliffe efficiency
        with absolute values to handle outliers
        """
        anse = 1 - (np.sum(np.abs(observed - modeled)) / np.sum(np.abs(observed - np.mean(observed))))
        return anse

    @staticmethod
    def nanse(observed: pd.Series, modeled: pd.Series) -> pd.Series:
        """
        Normalized modified version of Nash-Sutcliffe efficiency.
        :param pd.Series observed: The observed data. (validation)
        :param pd.Series modeled: The modeled data. (prediction)
        :return pd.Series: the normalized modified version of Nash-Sutcliffe efficiency
        """
        nanse = 1 - (np.sum(np.abs(observed - modeled)) / np.sum(np.abs(observed - np.mean(observed))))
        return 1 / (2 - nanse)

    @staticmethod
    def fc(observed: pd.Series, modeled: pd.Series) -> pd.Series:
        """
        Flow criteria
        :param pd.Series observed: The observed data. (validation)
        :param pd.Series modeled: The modeled data. (prediction)
        :return pd.Series: the calculated flow criteria
        """
        fc = ((np.sum((observed - modeled) ** 2 * observed ** 2)) ** 0.25) / (np.sum(observed ** 2) ** 0.5)
        return fc

    @staticmethod
    def lm(observed: pd.Series, modeled: pd.Series) -> pd.Series:
        """
        Legates and McCabe's (LM) index
        :param pd.Series observed: The observed data. (validation)
        :param pd.Series modeled: The modeled data. (prediction)
        :return pd.Series: the calculated Legates and McCabe's (LM) index
        """
        lm = 1 - (np.sum(np.abs(observed - modeled)) / np.sqrt(np.sum((observed - np.mean(observed)) ** 2)))
        return lm

    @staticmethod
    def wi(observed: pd.Series, modeled: pd.Series) -> pd.Series:
        """
        Willmott's index (WI)
        :param pd.Series observed: The observed data. (validation)
        :param pd.Series modeled: The modeled data. (prediction)
        :return pd.Series: the calculated Willmott's index (WI)
        """
        wi = 1 - (np.sum((observed - modeled) ** 2) /
                  np.sum(np.abs(observed - np.mean(observed)) + np.abs(modeled - np.mean(observed)) ** 2))
        return wi

    @staticmethod
    def nwi(observed: pd.Series, modeled: pd.Series) -> pd.Series:
        """
        Normalized Willmott's index (WI)
        :param pd.Series observed: The observed data. (validation)
        :param pd.Series modeled: The modeled data. (prediction)
        :return pd.Series: the calculated normalized Willmott's index (WI)
        """
        wi = 1 - (np.sum((observed - modeled) ** 2) /
                  np.sum(np.abs(observed - np.mean(observed)) + np.abs(modeled - np.mean(observed)) ** 2))
        return 1 / (2 - wi)

    # Basic similarity measures
    # MAPE
    @staticmethod
    def mape(observed: pd.Series, modeled: pd.Series) -> pd.Series:
        """
        Mean Absolute Percentage Error (MAPE)
        :param pd.Series observed: The observed data. (validation)
        :param pd.Series modeled: The modeled data. (prediction)
        :return pd.Series: the calculated  Mean Absolute Percentage Error (MAPE)
        """
        mape = np.mean(np.abs((observed - modeled) / observed)) * 100
        return mape

    # MAE
    @staticmethod
    def mae(observed: pd.Series, modeled: pd.Series) -> np.ndarray:
        """
        Mean Absolute Error (MAE)
        :param pd.Series observed: The observed data. (validation)
        :param pd.Series modeled: The modeled data. (prediction)
        :return np.ndarray: the calculated Mean Absolute Error (MAE)
        """
        mae = np.mean(np.absolute(np.subtract(observed, modeled)))
        return mae

    # RMAE
    @staticmethod
    def rmae(observed: pd.Series, modeled: pd.Series) -> pd.Series:
        """
        Root Mean Absolute Error (RMAE)
        :param pd.Series observed: The observed data. (validation)
        :param pd.Series modeled: The modeled data. (prediction)
        :return pd.Series: the calculated Root Mean Absolute Error (RMAE)
        """
        rmae = np.sqrt(np.mean(np.absolute(np.subtract(observed, modeled))))
        return rmae

    # MSE
    @staticmethod
    def mse(observed: pd.Series, modeled: pd.Series) -> np.ndarray:
        """
        Mean Square Error (MSE)
        :param pd.Series observed: The observed data. (validation)
        :param pd.Series modeled: The modeled data. (prediction)
        :return np.ndarray: the calculated Mean Square Error (MSE)
        """
        mse = np.mean((np.subtract(observed, modeled)) ** 2)
        return mse

    # RMSE
    @staticmethod
    def rmse(observed: pd.Series, modeled: pd.Series) -> pd.Series:
        """
        Root Mean Square Error (RMSE)
        :param pd.Series observed: The observed data. (validation)
        :param pd.Series modeled: The modeled data. (prediction)
        :return pd.Series: the calculated Root Mean Square Error (RMSE)
        """
        rmse = np.sqrt(np.mean((np.subtract(observed, modeled)) ** 2))
        return rmse

    # RRMSE
    @staticmethod
    def rrmse(observed: pd.Series, modeled: pd.Series) -> pd.Series:
        """
        Relative Root Mean Square Error (RRMSE)
        :param pd.Series observed: The observed data. (validation)
        :param pd.Series modeled: The modeled data. (prediction)
        :return pd.Series: the calculated Relative Root Mean Square Error (RRMSE)
        """
        rrmse = 100 * (np.sqrt(np.mean((np.subtract(observed, modeled)) ** 2)) / np.mean(observed))
        return rrmse

    # RMSRE
    @staticmethod
    def rmsre(observed: pd.Series, modeled: pd.Series) -> pd.Series:
        """
        Root Mean Square Relative Error (RMSRE)
        :param pd.Series observed: The observed data. (validation)
        :param pd.Series modeled: The modeled data. (prediction)
        :return pd.Series: the calculated Root Mean Square Relative Error (RMSRE)
        """
        rmsre = np.sqrt(np.mean((np.subtract(observed, modeled)) ** 2) / np.mean(observed))
        return rmsre

    @staticmethod
    def r2(observed: pd.Series, modeled: pd.Series) -> pd.Series:
        """
        R-squared (R2)
        :param pd.Series observed: The observed data. (validation)
        :param pd.Series modeled: The modeled data. (prediction)
        :return pd.Series: the calculated R-squared (R2)
        """
        return EvaluationBase.nse(observed, modeled)

    @staticmethod
    def corr(observed: pd.Series, modeled: pd.Series) -> pd.Series:
        """
        Calculate the correlation between the prediction and the real data based on the future.
        :param pd.Series observed: The observed data. (validation)
        :param pd.Series modeled: The modeled data. (prediction)
        :return pd.Series: the calculated correlation between the prediction and the real data based on the future.
        """
        corr = observed.corrwith(modeled, axis=0)
        return corr
