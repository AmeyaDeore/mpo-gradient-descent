import numpy as np
import pandas as pd


class DataManagement:
    def __init__(self):
        pass

    def get_data(asset_price_path: str, idx_price_path: str) -> tuple:
        """Gets the data of assets and the index.

        Args:
            asset_price_path (str): Path to the file with asset prices.
            idx_price_path (str): Path to the file with index prices.

        Returns:
            tuple: Data of assets and the index.
        """
        data = pd.read_csv(asset_price_path, parse_dates=True, index_col="Date", dtype=np.float32)
        data.dropna(inplace=True, axis=1, thresh=0.8 * data.shape[0])
        data.dropna(inplace=True, axis=0, how="any")

        data_idx = pd.read_csv(idx_price_path, parse_dates=True, index_col="Date", dtype=np.float32)
        data_idx.dropna(inplace=True)

        return data, data_idx

    def train_test_split_rolling(data: pd.DataFrame, w_train: int, w_test: int, fix_ini: bool = False):
        """
        Splits the data into training and test sets based on rolling windows.

        Parameters:
        data (pd.DataFrame): Data with dates in rows and assets in columns.
        w_train (int): Size of the training data window.
        w_test (int): Size of the test data window.
        fix_ini (bool): If True, training data always starts from the beginning (position 0).

        Returns:
        tuple: Two lists of DataFrames, the first with training data and the second with test data.
        """

        # Initialize lists to store training and test sets
        train_data = []
        test_data = []

        # Iterate over the data to create rolling windows
        for i in range(0, len(data) - w_train - w_test + 1, w_test):
            if fix_ini:
                # If fix_ini is True, the training window always starts from the beginning
                train_window = data.iloc[0 : i + w_train]
                test_window = data.iloc[i + w_train : i + w_train + w_test]
            else:
                # Create rolling windows for training and testing
                train_window = data.iloc[i : i + w_train, :]
                test_window = data.iloc[i + w_train : i + w_train + w_test, :]

            # Add the windows to the corresponding lists
            train_data.append(train_window)
            test_data.append(test_window)

        # Return the lists of training and test windows
        return train_data, test_data

    def train_split_rolling(X: pd.DataFrame, y: pd.DataFrame, window_size: int = 150, window_step: int = 1) -> tuple:
        """
        Prepares the data with rolling windows.

        Args:
        - X: DataFrame with columns of returns for each asset.
        - y: DataFrame with columns of returns for each index.
        - window_size: Size of the rolling window in training (default 150).
        - window_step: Step of the rolling window (default 1).

        Returns:
        - X_windows, y_windows: Tuple with lists of rolling windows for each set.
        """
        X_windows = []
        y_windows = []
        for i in range(0, len(X) - window_size + 1, window_step):
            X_window = X.iloc[i : i + window_size, :]
            y_window = y.iloc[i : i + window_size, :]
            X_windows.append(X_window)
            y_windows.append(y_window)

        return X_windows, y_windows

    def train_test_split_by_date(
        X: pd.DataFrame,
        y: pd.DataFrame,
        train_start_date: str,
        train_end_date: str,
        test_start_date: str,
        test_end_date: str,
        train_freq: str = "D",
    ) -> tuple:
        """Splits the data into TRAIN and TEST based on specified dates.

        Args:
            X (pd.DataFrame): Asset price data.
            y (pd.DataFrame): Index price data.
            train_start_date (str): Start date for TRAIN.
            train_end_date (str): End date for TRAIN.
            test_start_date (str): Start date for TEST.
            test_end_date (str): End date for TEST.
            train_freq (str, optional): Sampling frequency for TRAIN. Defaults to "D".
                Possible values: "D", "W", "2W", "M".

        Raises:
            ValueError: If TRAIN frequency is invalid. Must be "D", "W", "2W" or "M".

        Returns:
            tuple: TRAIN and TEST sets as DataFrames of returns.
        """
        # *******************************************************************************
        # Separation of the TRAIN set
        # *******************************************************************************
        if train_freq == "D":
            # DAILY DATA
            X_train = X.loc[train_start_date:train_end_date, :]
            y_train = y.loc[train_start_date:train_end_date, :]
        elif train_freq == "W":
            # WEEKLY DATA
            X_train = X.loc[train_start_date:train_end_date, :].resample("W-WED").last()
            y_train = y.loc[train_start_date:train_end_date, :].resample("W-WED").last()
        elif train_freq == "2W":
            # BIWEEKLY DATA
            X_train = X.loc[train_start_date:train_end_date, :].resample("2W-WED").last()
            y_train = y.loc[train_start_date:train_end_date, :].resample("2W-WED").last()
        elif train_freq == "M":
            # MONTHLY DATA
            X_train = X.loc[train_start_date:train_end_date, :].resample("M").last()
            y_train = y.loc[train_start_date:train_end_date, :].resample("M").last()
        else:
            raise ValueError("train_freq must be 'D', 'W', '2W' or 'M'.")

        # *******************************************************************************
        # Separation of the TEST set
        # *******************************************************************************
        X_test = X.loc[test_start_date:test_end_date, :]
        y_test = y.loc[test_start_date:test_end_date, :]

        return X_train, y_train, X_test, y_test

    def convert_prices_to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Converts prices to logarithmic returns.

        Args:
            prices (pd.DataFrame): DataFrame with prices.

        Returns:
            pd.DataFrame: DataFrame with logarithmic returns.
        """
        return np.log(prices).diff().dropna()

    def sync_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple:
        """Synchronizes two DataFrames by common dates.

        Args:
            df1 (pd.DataFrame): First DataFrame.
            df2 (pd.DataFrame): Second DataFrame.

        Returns:
            tuple: The two synchronized DataFrames.
        """
        df1_index = df1.index
        df2_index = df2.index
        common_index = df1_index.intersection(df2_index)

        df1 = df1.loc[common_index]
        df2 = df2.loc[common_index]

        return df1, df2
