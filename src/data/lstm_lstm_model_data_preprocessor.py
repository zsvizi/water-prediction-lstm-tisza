import datetime
import json
import os
from typing import Any, Tuple

import numpy as np
import pandas as pd
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import EncoderNormalizer

from src import PROJECT_PATH
from src.data.data_preprocessor import DataPreprocessor
from src.data.downloader import Downloader


class LSTMLSTMModelDataPreprocessor:
    """
    A class to manage data manipulations for the LSTM-LSTM model.
    """
    def __init__(self):
        model_name = "t19"
        Downloader(gdrive_id="1MnTYl60N6sGnD0AMeqcrBOHigEKjEIkp",
                   file_name=model_name + ".json")
        model_files = json.load(open(os.path.join(PROJECT_PATH, "data", model_name + ".json")))
        Downloader(gdrive_id=model_files[model_name]["hparams"], file_name="hparams_" + model_name + ".yaml")
        Downloader(gdrive_id=model_files[model_name]["parameter"], file_name="parameter_file_" + model_name)

    @staticmethod
    def get_dataloaders(data, scalers, train, max_encoder_length, max_prediction_length, features, batch_size,
                        target_normalizer=EncoderNormalizer(),
                        num_workers: int = 0, target: str = None) -> Tuple[Any, Any]:
        dataset = TimeSeriesDataSet(data, time_idx="time_idx", target=target, group_ids=["group_id"],
                                    min_encoder_length=max_encoder_length, max_encoder_length=max_encoder_length,
                                    min_prediction_length=max_prediction_length,
                                    max_prediction_length=max_prediction_length, time_varying_known_reals=[],
                                    time_varying_unknown_reals=features, scalers=scalers,
                                    target_normalizer=target_normalizer)
        dataloader = dataset.to_dataloader(train=train, batch_size=batch_size, num_workers=num_workers)
        return dataset, dataloader

    @staticmethod
    def preprocess_data(df: pd.DataFrame, start_date: str, end_date: str, hyperparameters: dict):
        """
        This method preprocesses the data.
        :param pd.DataFrame df: the data that will be preprocessed
        :param start_date: start date of the appropriate time interval (included)
        :param end_date: end date of the appropriate time interval (included)
        :param dict hyperparameters: hyperparameters of the data
        :return pd.DataFrame result: the preprocessed data
        """
        hp = hyperparameters
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
    def extend_df_for_predictions_after_last_days(data: pd.DataFrame, end_date_temp, last_date,
                                                  result: pd.DataFrame) -> pd.DataFrame:
        """
        This method extends the data for predictions after last days.
        :param pd.DataFrame data: the data that will be extended
        :param end_date_temp: the temporary end date
        :param last_date: the last date
        :param pd.DataFrame result: the data that will be extended
        :return pd.DataFrame result: the extended dataframe
        """
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
