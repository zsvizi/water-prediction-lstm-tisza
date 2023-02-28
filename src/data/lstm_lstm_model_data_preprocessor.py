import datetime
import json
import os

import numpy as np
import pandas as pd
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import EncoderNormalizer

from src import PROJECT_PATH
from src.data.data_preprocessor import DataPreprocessor
from src.data.downloader import Downloader


class LSTMLSTMModelDataPreprocessor:
    def __int__(self):
        pass

    @staticmethod
    def get_model_files(model_name: str):
        Downloader(file_url="https://drive.google.com/uc?export=download&id=1JqH05bnS9Q2RM2MGzTPf2obDIaX9eM3J",
                   file_name="modellek.json")
        data = json.load(open(os.path.join(PROJECT_PATH, "data", "modellek.json")))

        Downloader(file_url=data[model_name]["hparams"], file_name="hparams_" + model_name + ".yaml")
        Downloader(file_url=data[model_name]["parameter"], file_name="parameter_file_" + model_name)

    @staticmethod
    def get_dataloaders(data, scalers, train, max_encoder_length, max_prediction_length, features, batch_size,
                        target_normalizer=EncoderNormalizer(), num_workers: int = 0, target: str = None):
        dataset = TimeSeriesDataSet(data, time_idx="time_idx", target=target, group_ids=["group_id"],
                                    min_encoder_length=max_encoder_length, max_encoder_length=max_encoder_length,
                                    min_prediction_length=max_prediction_length,
                                    max_prediction_length=max_prediction_length, time_varying_known_reals=[],
                                    time_varying_unknown_reals=features, scalers=scalers,
                                    target_normalizer=target_normalizer)
        dataloader = dataset.to_dataloader(train=train, batch_size=batch_size, num_workers=num_workers)
        return dataset, dataloader

    @staticmethod
    def load_data(start_date, end_date, hyperparameters):
        hp = hyperparameters
        Downloader(file_url="https://drive.google.com/uc?export=download&id=1MXUseGykD-Tf1cAJ9Ipp-vYJISNwyCy3",
                   file_name="data_2004-2020.csv")
        df = pd.read_csv(os.path.join(PROJECT_PATH, "data", "data_2004-2020.csv"), index_col=0)
        df.columns = df.columns.astype(str)
        dm = DataPreprocessor(df)

        start_date_temp = datetime.datetime.strptime(start_date, "%Y-%m-%d") \
            + datetime.timedelta(days=-hp["max_encoder_length"] + 1)
        end_date_temp = datetime.datetime.strptime(end_date, "%Y-%m-%d") \
            + datetime.timedelta(days=hp["max_prediction_length"] + 1)

        result = dm.filter_by_dates(start_date_temp, end_date_temp)

        last_date = pd.to_datetime(df.index[-1])
        if end_date_temp > last_date:
            result = LSTMLSTMModelDataPreprocessor.extend_df_for_predictions_after_last_days(
                data=df, end_date_temp=end_date_temp, last_date=last_date, result=result)

        return result

    @staticmethod
    def extend_df_for_predictions_after_last_days(data, end_date_temp, last_date, result):
        # add last date to the filtered data (since it does not contain)
        to_append = pd.DataFrame(
            data=np.hstack((
                data.iloc[-1].values,
                [0], [0], [0]
            )).reshape((1, -1)),
            columns=result.columns,
            index=[data.index[-1]])
        result = pd.concat((result, to_append))
        result["group_id"].iloc[-1] = result["group_id"].iloc[-2]
        result["time_idx"].iloc[-1] = result["time_idx"].iloc[-2] + 1
        result["day"].iloc[-1] = result["day"].iloc[-2] + 1
        # create array with dummy values for the time series
        # create valid values for time_idx and day
        day_diff = int((end_date_temp - last_date).days) - 1
        fill_array = np.hstack((
            np.zeros((day_diff, len(result.columns) - 2)),
            np.arange(result["time_idx"].iloc[-1] + 1,
                      result["time_idx"].iloc[-1] + 1 + day_diff
                      ).reshape((-1, 1)),
            np.arange(result["day"].iloc[-1] + 1,
                      result["day"].iloc[-1] + 1 + day_diff
                      ).reshape((-1, 1))
        )).astype(int)
        # create dataframe to concatenate
        df_concat = pd.DataFrame(
            data=fill_array,
            index=pd.date_range(pd.to_datetime(result.index[-1]) + datetime.timedelta(days=1),
                                end_date_temp - datetime.timedelta(days=1)),
            columns=result.columns)
        # concatenate dummy values to the resulting dataframe
        result = pd.concat((result, df_concat))
        result = result.astype({"time_idx": int}, errors="ignore")
        return result
