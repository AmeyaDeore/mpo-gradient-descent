import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
from typing import List, Tuple

class DataLoader:
    """Loads and prepares market data for optimization."""

    DATA_PATH = Path(__file__).parent.parent / "data"
    ASSET_FILE = DATA_PATH / "data_comp_SP500.csv"
    INDEX_FILE = DATA_PATH / "data_idx_SP500.csv"

    def __init__(self):
        """Initialize data loader."""
        self.asset_data = None
        self.index_data = None

    @st.cache_data
    def load_data(_self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load asset and index data with caching.

        Returns:
            Tuple of (asset_data, index_data) DataFrames
        """
        try:
            # Load asset prices
            asset_data = pd.read_csv(
                _self.ASSET_FILE,
                parse_dates=True,
                index_col="Date",
                dtype=np.float32
            )

            # Load index prices
            index_data = pd.read_csv(
                _self.INDEX_FILE,
                parse_dates=True,
                index_col="Date",
                dtype=np.float32
            )

            return asset_data, index_data

        except FileNotFoundError as e:
            raise Exception(f"Data file not found: {str(e)}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def filter_and_align(self, assets_df: pd.DataFrame,
                        tickers: List[str]) -> pd.DataFrame:
        """
        Filter assets and ensure data quality.

        Args:
            assets_df: Full asset data DataFrame
            tickers: List of selected tickers

        Returns:
            Filtered and cleaned DataFrame
        """
        # Filter to selected tickers
        filtered = assets_df[[t for t in tickers if t in assets_df.columns]].copy()

        # Remove rows with NaN
        filtered = filtered.dropna()

        # Ensure we have enough data
        if len(filtered) < 100:
            raise ValueError(f"Insufficient data: only {len(filtered)} observations")

        return filtered

    def convert_to_returns(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert prices to log returns.

        Args:
            prices_df: DataFrame of prices

        Returns:
            DataFrame of log returns
        """
        log_prices = np.log(prices_df)
        returns = log_prices.diff().dropna()
        return returns

    def load_and_prepare_data(self, tickers: List[str]) -> pd.DataFrame:
        """
        Load, filter, and convert data to returns.

        Args:
            tickers: List of stock tickers to load

        Returns:
            DataFrame of daily log returns
        """
        # Load raw data
        asset_data, _index_data = self.load_data()

        # Filter to selected tickers
        filtered_data = self.filter_and_align(asset_data, tickers)

        # Convert to returns
        returns_data = self.convert_to_returns(filtered_data)

        return returns_data

    def get_statistics(self, returns_df: pd.DataFrame) -> dict:
        """
        Calculate basic statistics for returns.

        Args:
            returns_df: DataFrame of returns

        Returns:
            Dictionary of statistics
        """
        ann_factor = 252  # Trading days per year

        return {
            'mean_return': returns_df.mean() * ann_factor,
            'std_dev': returns_df.std() * np.sqrt(ann_factor),
            'correlation': returns_df.corr(),
            'covariance': returns_df.cov() * ann_factor
        }
