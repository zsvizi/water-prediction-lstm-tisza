import pandas as pd


class DataPreprocessor:
    """
    A class to manage data manipulations.
    """
    def __init__(self, dataframe: pd.DataFrame):
        """
        :param pd.DataFrame dataframe: the index of the input dataframe has to be DateTimeIndex
        """
        self.data = dataframe

    @staticmethod
    def add_forecasting_columns(data: pd.DataFrame) -> pd.DataFrame:
        """
        Add extra columns to the input dataframe, that are required by the PyTorchForecasting framework.
        :param pd.DataFrame data: pandas dataframe, whose index is a datetime index
        :return pd.DataFrame df: a new dataframe with additional columns:
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
        Creates a dataframe that is filtered and additional columns are added.
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
