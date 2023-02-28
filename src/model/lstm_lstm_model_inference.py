import datetime
import os

import numpy as np
import pandas as pd

from src import PROJECT_PATH
from src.data.downloader import Downloader
from src.data.data_preprocessor import DataPreprocessor
from src.data.lstm_lstm_model_data_preprocessor import LSTMLSTMModelDataPreprocessor


class LSTMLSTMModelInference:
    def __init__(self, model, p_yaml_dict: dict, hyperparameters):
        self.model = model
        self.p_yaml_dict = p_yaml_dict
        self.hyperparameters = hyperparameters

    def get_prediction(self, start_date, end_date):

        hp = self.hyperparameters

        pred_length = hp['max_prediction_length']

        conf_calc_window = 14
        conf_start = datetime.datetime.strftime(datetime.datetime.strptime(start_date, '%Y-%m-%d') -
                                                datetime.timedelta(days=conf_calc_window), '%Y-%m-%d')

        Downloader(file_url="https://drive.google.com/uc?export=download&id=1MXUseGykD-Tf1cAJ9Ipp-vYJISNwyCy3",
                   file_name="data_2004-2020.csv")
        df = pd.read_csv(os.path.join(PROJECT_PATH, "data", "data_2004-2020.csv"), index_col=0)
        df.columns = df.columns.astype(str)
        dm = DataPreprocessor(df)

        _, d_val_dataloader_conf = LSTMLSTMModelDataPreprocessor.get_dataloaders(
            data=LSTMLSTMModelDataPreprocessor.load_data(start_date=conf_start, end_date=end_date, hyperparameters=hp),
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
