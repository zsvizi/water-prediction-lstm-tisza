from itertools import chain
import operator
import os
from typing import Any, List, Tuple

import numpy as np
import pandas as pd

from src import PROJECT_PATH
from src.data.downloader import Downloader


class DataPreprocessor:
    """
    A class to manage data manipulations
    """
    def __init__(self, dataframe: pd.DataFrame):
        """
        :param dataframe: The index of the input dataframe has to be DateTimeIndex
        """
        self.data = dataframe

    @staticmethod
    def add_forecasting_columns(data: pd.DataFrame) -> pd.DataFrame:
        """
        Add extra columns to the input dataframe, that are required by the PyTorchForecasting framework

        :param data: pandas dataframe, whose index is a datetime index
        :return: a new dataframe with additional columns:
            - group_id: ID, that corresponds to the time series
            - time_idx: ID of the element, in the corresponding time series
            - day: number of the day in the year
        """
        df = data.copy()
        df.dropna(inplace=True)
        df.reset_index(inplace=True)
        df['index'] = pd.to_datetime(df['index'])
        df['group_id'] = df["index"].diff().dt.days.ne(1).cumsum() - 1
        df['time_idx'] = df.groupby('group_id').cumcount(ascending=True)
        df['day'] = df['index'].dt.dayofyear
        df.set_index("index", inplace=True)
        return df

    def filter_by_dates(self, start_date, end_date) -> pd.DataFrame:
        """
        Select
        :param start_date: start date of the appropriate time interval (included)
        :param end_date: end date of the appropriate time interval (included)
        :return: pd.DataFrame a new pandas dataframe that is filtered and additional columns are added
        """
        df = self.data.copy()
        if type(start_date) == str:
            pass
        else:
            start_date = start_date.strftime("%Y-%m-%d")

        if type(end_date) == str:
            pass
        else:
            end_date = end_date.strftime("%Y-%m-%d")
        df_filtered = df.loc[start_date:end_date]
        df_filtered.sort_index(inplace=True)
        return self.add_forecasting_columns(df_filtered[:-1])

    @staticmethod
    def format_df(df: pd.DataFrame, d_test_data_end: str) -> pd.DataFrame:
        df.columns = df.columns.astype(str)
        dm = DataPreprocessor(df)

        d_full = dm.filter_by_dates('1900-01-01', d_test_data_end)
        return d_full

    @staticmethod
    def create_gauge(gauge_row, gauge_col, start, end):
        gauge_row_y = DataPreprocessor.crop_by_period(gauge_row, start, end)
        gauge_col_y = DataPreprocessor.crop_by_period(gauge_col, start, end)
        return gauge_row_y, gauge_col_y

    @staticmethod
    def load_data():
        Downloader(file_url="https://drive.google.com/uc?export=download&id=1MXUseGykD-Tf1cAJ9Ipp-vYJISNwyCy3",
                   file_name="data_2004-2020.csv")
        df = pd.read_csv(os.path.join(PROJECT_PATH, "data", "data_2004-2020.csv"), index_col=0)
        return df

    @staticmethod
    def crop_by_period(data: pd.DataFrame, start: str, end: str = None) -> pd.DataFrame:
        data = data[data.index >= start]
        data = data[data.index <= end]
        return data

    @staticmethod
    def higlight_rows_list(df: pd.DataFrame, features, colors):
        for idx in range(len(df)):
            if df.index[idx] in features:
                colors[idx] = ['lightgreen'] * len(df.columns)
        return colors

    @staticmethod
    def normalize(df):
        max_value = df.max().max()
        return df / max_value

    @staticmethod
    def symmetrize_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
        """
        :param pd.DataFrame matrix: The matrix, we want to symmetrize.
        :return: The symmetric matrix
        """
        values = (matrix.to_numpy() + matrix.T.to_numpy()) / 2
        mean_matrix = pd.DataFrame(values, index=matrix.columns, columns=matrix.columns)
        return mean_matrix

    @staticmethod
    def rearrange_matrix(cluster, header, medoids, matrix_value) -> Tuple[np.ndarray, List[Any], Any]:
        """
        Rearrange the correlation matrix, based on the hierarchical clustering, sorting clusters by size.
        :return pd.DataFrame: The rearranged matrix
        """
        original = list(zip(cluster, header))
        all_clusters = [[] for _ in range(medoids)]
        original = np.array(sorted(original, key=operator.itemgetter(0)))
        clusters = [0]
        for value in range(1, medoids + 1):
            local = np.count_nonzero(cluster == value)
            clusters.append(local)
            cum_sum = np.cumsum(clusters)
            new_cluster = original[cum_sum[value - 1]: cum_sum[value]]
            all_clusters[value - 1] = new_cluster[:, 1]
        all_clusters = sorted(all_clusters, key=len, reverse=True)
        all_clusters = list(chain.from_iterable(all_clusters))
        rearranged = matrix_value.loc[all_clusters, all_clusters]

        return original, all_clusters, rearranged
