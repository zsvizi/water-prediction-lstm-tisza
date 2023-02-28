import numpy as np


class EvaluationBase:
    def __init__(self):
        pass

    # Nash-Sutcliffe efficiency NSE
    @staticmethod
    def nse(observed, modeled):
        """
        Calculate the Nash-Sutcliffe efficiency.
        :param pd.Series observed: The observed data. (validation)
        :param pd.Series modeled: The modeled data. (prediction)
        """
        nse = 1 - (np.sum((observed - modeled) ** 2) / np.sum((observed - np.mean(observed)) ** 2))
        return nse

    @staticmethod
    def nnse(observed, modeled):
        """
        Calculate the normalized Nash-Sutcliffe efficiency.
        :param pd.Series observed: The observed data. (validation)
        :param pd.Series modeled: The modeled data. (prediction)
        """
        nse = 1 - (np.sum((observed - modeled) ** 2) / np.sum((observed - np.mean(observed)) ** 2))
        return 1 / (2 - nse)

    @staticmethod
    def anse(observed, modeled):
        """
        Calculate a modified version of Nash-Sutcliffe efficiency with absolute values to handle outliers.
        :param pd.Series observed: The observed data. (validation)
        :param pd.Series modeled: The modeled data. (prediction)
        """
        anse = 1 - (np.sum(np.abs(observed - modeled)) / np.sum(np.abs(observed - np.mean(observed))))
        return anse

    @staticmethod
    def nanse(observed, modeled):
        """
        Normalized modified version of Nash-Sutcliffe efficiency.
        :param pd.Series observed: The observed data. (validation)
        :param pd.Series modeled: The modeled data. (prediction)
        """
        anse = 1 - (np.sum(np.abs(observed - modeled)) / np.sum(np.abs(observed - np.mean(observed))))
        return 1 / (2 - anse)

    @staticmethod
    def fc(observation, modeled):
        """
        Flow criteria
        :param pd.Series observation: The observed data. (validation)
        :param pd.Series modeled: The modeled data. (prediction)
        """
        fc = ((np.sum((observation - modeled) ** 2 * observation ** 2)) ** 0.25) / (np.sum(observation ** 2) ** 0.5)
        return fc

    @staticmethod
    def lm(observed, modeled):
        """
        Legates and McCabe's (LM) index
        :param pd.Series observed: The observed data. (validation)
        :param pd.Series modeled: The modeled data. (prediction)
        """
        lm = 1 - (np.sum(np.abs(observed - modeled)) / np.sqrt(np.sum((observed - np.mean(observed)) ** 2)))
        return lm

    @staticmethod
    def wi(observed, modeled):
        """
        Willmott's index (WI)
        :param pd.Series observed: The observed data. (validation)
        :param pd.Series modeled: The modeled data. (prediction)
        """
        wi = 1 - (np.sum((observed - modeled) ** 2) /
                  np.sum(np.abs(observed - np.mean(observed)) + np.abs(modeled - np.mean(observed)) ** 2))
        return wi

    @staticmethod
    def nwi(observed, modeled):
        """
        Normalized Willmott's index (WI)
        :param pd.Series observed: The observed data. (validation)
        :param pd.Series modeled: The modeled data. (prediction)
        """
        wi = 1 - (np.sum((observed - modeled) ** 2) /
                  np.sum(np.abs(observed - np.mean(observed)) + np.abs(modeled - np.mean(observed)) ** 2))
        return 1 / (2 - wi)

    # Basic similarity measures
    # MAPE
    @staticmethod
    def mape(observed, modeled):
        """
        Mean Absolute Percentage Error (MAPE)
        :param pd.Series observed: The observed data. (validation)
        :param pd.Series modeled: The modeled data. (prediction)
        """
        mape = np.mean(np.abs((observed - modeled) / observed)) * 100
        return mape

    # MAE
    @staticmethod
    def mae(observed, modeled):
        """
        Mean Absolute Error (MAE)
        :param pd.Series observed: The observed data. (validation)
        :param pd.Series modeled: The modeled data. (prediction)
        """
        mae = np.mean(np.absolute(np.subtract(observed, modeled)))
        return mae

    # RMAE
    @staticmethod
    def rmae(observed, modeled):
        """
        Root Mean Absolute Error (RMAE)
        :param pd.Series observed: The observed data. (validation)
        :param pd.Series modeled: The modeled data. (prediction)
        """
        rmae = np.sqrt(np.mean(np.absolute(np.subtract(observed, modeled))))
        return rmae

    # MSE
    @staticmethod
    def mse(observed, modeled):
        """
        Mean Square Error (MSE)
        :param pd.Series observed: The observed data. (validation)
        :param pd.Series modeled: The modeled data. (prediction)
        """
        mse = np.mean((np.subtract(observed, modeled)) ** 2)
        return mse

    # RMSE
    @staticmethod
    def rmse(observed, modeled):
        """
        Root Mean Square Error (RMSE)
        :param pd.Series observed: The observed data. (validation)
        :param pd.Series modeled: The modeled data. (prediction)
        """
        rmse = np.sqrt(np.mean((np.subtract(observed, modeled)) ** 2))
        return rmse

    # RRMSE
    @staticmethod
    def rrmse(observed, modeled):
        """
        Relative Root Mean Suare Error (RRMSE)
        :param pd.Series observed: The observed data. (validation)
        :param pd.Series modeled: The modeled data. (prediction)
        """
        rrmse = 100 * (np.sqrt(np.mean((np.subtract(observed, modeled)) ** 2)) / np.mean(observed))
        return rrmse

    # RMSRE
    @staticmethod
    def rmsre(observed, modeled):
        """
        Root Mean Square Relative Error (RMSRE)
        :param pd.Series observed: The observed data. (validation)
        :param pd.Series modeled: The modeled data. (prediction)
        """
        rmsre = np.sqrt(np.mean((np.subtract(observed, modeled)) ** 2) / np.mean(observed))
        return rmsre

    @staticmethod
    def r2(observed, modeled):
        """
        R-squared (R2)
        :param pd.Series observed: The observed data. (validation)
        :param pd.Series modeled: The modeled data. (prediction)
        """
        return EvaluationBase.nse(observed, modeled)

    @staticmethod
    def corr(observed, modeled):
        """
        Calculate the correlation between the prediction and the real data based on the future.
        :param pd.Series observed: The observed data. (validation)
        :param pd.Series modeled: The modeled data. (prediction)
        """
        corr = observed.corrwith(modeled, axis=0)
        return corr
