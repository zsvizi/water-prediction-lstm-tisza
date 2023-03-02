import datetime

import numpy as np
import pandas as pd

from src.data.data_preprocessor import DataPreprocessor
from src.data.lstm_lstm_model_data_preprocessor import LSTMLSTMModelDataPreprocessor


class LSTMLSTMModelInference:
    def __init__(self, model, p_yaml_dict: dict, hyperparameters: dict):
        """
        The LSTM-LSTM model inference class.
        :param model: the model
        :param dict p_yaml_dict: the yaml dict.
        :param dict hyperparameters: hyperparameters of the model
        """
        self.model = model
        self.p_yaml_dict = p_yaml_dict
        self.hyperparameters = hyperparameters

    def get_prediction(self, df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """
        This method predicts data with LSTM-LSTM model.
        :param pd.DataFrame df: the dataframe, that will be predicted
        :param str start_date: start date of the appropriate time interval (included)
        :param str end_date: end date of the appropriate time interval (included)
        :return pd.DataFrame df_result: the predicted data
        """
        hp = self.hyperparameters

        pred_length = hp['max_prediction_length']

        conf_calc_window = 14
        conf_start = datetime.datetime.strftime(datetime.datetime.strptime(start_date, '%Y-%m-%d') -
                                                datetime.timedelta(days=conf_calc_window), '%Y-%m-%d')

        df.columns = df.columns.astype(str)
        dm = DataPreprocessor(df)

        _, d_val_dataloader_conf = LSTMLSTMModelDataPreprocessor.get_dataloaders(
            data=LSTMLSTMModelDataPreprocessor.preprocess_data(
                df=df, start_date=conf_start, end_date=end_date, hyperparameters=hp),
            train=False, scalers=hp["scalers"],
            max_encoder_length=hp["max_encoder_length"],
            max_prediction_length=pred_length,
            features=hp["features"], batch_size=1, target=hp["target"],
            target_normalizer=hp["normalizer"])

        m_raw_predictions_conf, _ = self.model.predict(d_val_dataloader_conf, mode="raw", return_x=True)

        filter_end = datetime.datetime.strptime(end_date, '%Y-%m-%d') + datetime.timedelta(days=pred_length)

        conf_start = datetime.datetime.strptime(conf_start, '%Y-%m-%d')

        data_to_conf_calc = dm.filter_by_dates(conf_start, filter_end)

        last_date = df.index[-1]
        last_date = datetime.datetime.strptime(last_date, "%Y-%m-%d")

        if filter_end > last_date:
            df_append = pd.DataFrame(
                data=[df.iloc[-1].values],
                columns=data_to_conf_calc.columns[:-3],
                index=[df.index[-1]])
            data_to_conf_calc = pd.concat((data_to_conf_calc, df_append))

        start_date_new = datetime.datetime.strftime(datetime.datetime.strptime(start_date, '%Y-%m-%d') +
                                                    datetime.timedelta(days=1), '%Y-%m-%d')
        end_date_new = datetime.datetime.strftime(datetime.datetime.strptime(end_date, '%Y-%m-%d') +
                                                  datetime.timedelta(days=1), '%Y-%m-%d')
        dt = pd.date_range(start_date_new, end_date_new)
        df_result = pd.DataFrame(m_raw_predictions_conf[0][conf_calc_window:, :, :].numpy()
                                 .reshape((len(dt), pred_length)))
        df_result.index = dt
        df_result.index.name = 'Date'

        dd = data_to_conf_calc.loc[:, hp["target"]].to_numpy().reshape((-1, 1))
        real_data_conf = np.hstack([np.roll(dd, -i)[:m_raw_predictions_conf.prediction.shape[0]]
                                    for i in range(pred_length)])
        diff = np.fabs(m_raw_predictions_conf[0].numpy().reshape((-1, pred_length)) - real_data_conf)
        conf = []
        for i in range(len(dt)):
            conf.append(np.std(diff[i:i + conf_calc_window, :], axis=0))

        original = df_result.to_numpy()
        df_result[['lower_' + str(i) for i in range(pred_length)]] = original - np.array(conf)
        df_result[['upper_' + str(i) for i in range(pred_length)]] = original + np.array(conf)

        return df_result
