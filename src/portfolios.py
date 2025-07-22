from pydantic import BaseModel

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns

from risk_measures import RiskMeasures


class Portfolio(BaseModel):
    name: str
    asset_weights: np.ndarray
    asset_names: list[str]

    class Config:
        arbitrary_types_allowed = True

    def compute_returns(self, asset_returns: np.ndarray) -> pd.Series:
        """Calculates portfolio returns.

        Args:
            asset_returns (np.ndarray): Asset returns.

        Returns:
            pd.Series: Portfolio returns.
        """
        p_ret = np.dot(asset_returns, self.asset_weights)
        return pd.Series(p_ret, name=self.name, index=asset_returns.index)

    def plot_weights(
        self,
        plot_ucits_limits=True,
        plot_min_weight=False,
        min_weight=0.0,
        skip_zero_weights=False,
        zero_threshold: float = 0.0,
        save_as: str = None,
        figsize=(8, 4),
    ) -> None:
        """Plots portfolio weights.

        Args:
            plot_ucits_limits (bool, optional): Displays UCITS limits. Defaults to True.
            plot_min_weight (bool, optional): Displays minimum weight. Defaults to False.
            min_weight (float, optional): Minimum weight. Defaults to 0.0.
            skip_zero_weights (bool, optional): Whether to skip zero weights. Defaults to False.
            zero_threshold (float, optional): Threshold to consider a weight as zero. Defaults to 0.0.
            save_as (str, optional): Path to save the image. Defaults to None.
            figsize (tuple, optional): Figure size. Defaults to (8, 4).
        """
        df = pd.DataFrame(self.asset_weights, index=self.asset_names, columns=["Weight"])

        # Set to zero using threshold.
        mask = df["Weight"] < zero_threshold
        df[mask] = 0.0

        if skip_zero_weights:
            mask = df["Weight"] > 0.0
            df = df[mask]

        if plot_ucits_limits:
            mask = np.where(self.asset_weights > 0.05, 1, 0)
            print(f"Sum of weights > 5%: {np.sum(self.asset_weights * mask)}")

        mask = np.where(self.asset_weights > 0.0, 1, 0)
        print(f"Number of assets in portfolio: {np.sum(mask)}")

        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(x=df.index, y=df["Weight"], ax=ax)

        if plot_ucits_limits:
            plt.axhline(0.05, color="r", linestyle="--", label="UCITS 5% Limit")
            plt.axhline(0.10, color="r", linestyle="--", label="UCITS 10% Limit")
            plt.legend()

        if plot_min_weight:
            plt.axhline(min_weight, color="g", linestyle="--", label="Min Weight")
            plt.legend()

        if skip_zero_weights is False:
            plt.xticks([])
        else:
            plt.xticks(df.index, rotation=45, ha="right")

        plt.xlabel("Assets", fontdict={"fontsize": 10})
        plt.ylabel("Weight", fontdict={"fontsize": 10})
        plt.title("Portfolio Weights", fontdict={"fontsize": 15})
        plt.tight_layout()

        if save_as is not None:
            plt.savefig(save_as)

        plt.show()

        print(df.T.to_latex())

    def compute_metrics(self, asset_ret: np.ndarray, idx_ret: np.ndarray, rf: float = 0.0) -> dict:
        """Calculates portfolio metrics and metrics relative to the index.

        Args:
            asset_ret (np.ndarray): Asset returns.
            idx_ret (np.ndarray): Index returns.
            rf (float): Risk-free rate.

        Returns:
            dict: sharpe, tracking_error, var, cvar, weights over 5%.
        """
        p_ret = self.compute_returns(asset_ret).to_numpy()

        s = RiskMeasures.Sharpe_Ratio(p_ret, rf)
        te = RiskMeasures.Tracking_Error(idx_ret, p_ret)
        var = RiskMeasures.VaR_Hist(p_ret, alpha=0.05)
        cvar = RiskMeasures.CVaR_Hist(p_ret, alpha=0.05)
        std = RiskMeasures.Std(p_ret)

        # Weights exceeding 10%
        mask = np.where(self.asset_weights > 0.10, True, False)
        weights_010 = np.sum(self.asset_weights[mask] - 0.10)

        # Weights exceeding 5%
        mask = np.where(self.asset_weights > 0.05, 1, 0)
        weights_005 = np.sum(self.asset_weights * mask)

        return {
            "Std": std,
            "Sharpe": s,
            "TrackingError": te,
            "VaR": var,
            "CVaR": cvar,
            "WeightsOver10pct": weights_010,
            "WeightsOver5pct": weights_005,
        }

    def compute_idx_metrics(self, idx_ret: np.ndarray, rf: float = 0.0) -> dict:
        """Calculates index metrics.

        Args:
            idx_ret (np.ndarray): Index returns.
            rf (float): Risk-free rate.

        Returns:
            dict: sharpe, tracking_error, var, cvar, weights over 5%.
        """
        s = RiskMeasures.Sharpe_Ratio(idx_ret, rf)
        var = RiskMeasures.VaR_Hist(idx_ret, alpha=0.05)
        cvar = RiskMeasures.CVaR_Hist(idx_ret, alpha=0.05)
        std = RiskMeasures.Std(idx_ret)

        return {
            "Std": std,
            "Sharpe": s,
            "TrackingError": 0.0,
            "VaR": var,
            "CVaR": cvar,
            "WeightsOver10pct": np.nan,
            "WeightsOver5pct": np.nan,
        }


class Portfolio_Collection:
    portfolios: list[str] = []
    asset_names: list[str] = []

    def __init__(self, portfolio_names: list, portfolio_weights: list, asset_names: list):
        """Constructor for the Portfolios class.

        Args:
            portfolio_names (list): Portfolio names.
            portfolio_weights (list): Portfolio weights.
            asset_names (list): Asset names.
        """
        self.portfolios: list = [Portfolio(n, w, asset_names) for n, w in zip(portfolio_names, portfolio_weights)]
        self.asset_names: list = asset_names

    def get_portfolio_names(self) -> list:
        """Gets portfolio names.

        Returns:
            list: Portfolio names.
        """
        return [p.name for p in self.portfolios]

    def get_portfolio_weights(self) -> list:
        """Gets portfolio weights.

        Returns:
            list: Portfolio weights.
        """
        return [p.asset_weights for p in self.portfolios]

    def append_portfolio(self, portfolio: Portfolio) -> None:
        """Adds a portfolio to the class.

        Args:
            portfolio (Portfolio): Portfolio to add.
        """
        self.portfolios.append(portfolio)

    def remove_portfolio(self, name: str) -> None:
        """Removes a portfolio from the class.

        Args:
            name (str): Name of the portfolio to remove.
        """
        self.portfolios = [p for p in self.portfolios if p.name != name]

    def replace_portfolio(self, name: str, new_portfolio: Portfolio) -> None:
        """Replaces a portfolio in the class.

        Args:
            name (str): Name of the portfolio to replace.
            new_portfolio (Portfolio): New portfolio.
        """
        for i, existing_portfolio in enumerate(self.portfolios):
            if existing_portfolio.name == name:
                self.optimizers[i] = new_portfolio
                return

        raise ValueError(f"Portfolio '{name}' not found.")

    def replace_or_add_portfolio(self, name: str, new_portfolio: Portfolio) -> None:
        """Replaces or adds a portfolio to the class.

        Args:
            name (str): Name of the portfolio to replace or add.
            new_portfolio (Portfolio): New portfolio.
        """
        for i, existing_portfolio in enumerate(self.portfolios):
            if existing_portfolio.name == name:
                self.portfolios[i] = new_portfolio
                return
        self.portfolios.append(new_portfolio)

    def plot_cumulative_returns(
        self,
        asset_returns: pd.DataFrame,
        idx_returns: pd.Series,
        portfolio_names: list,
        plot_idx: bool = True,
        comparable_returns: list = [],
        rf: float = 0.0,
        zero_start: bool = False,
        save_as: str = None,
        figsize: tuple = (8, 4),
    ) -> None:
        """Plots accumulated returns for portfolios and the index along with other comparables,
        printing metrics to the console.

        Args:
            asset_returns (pd.DataFrame): Asset returns.
            idx_returns (pd.Series): Index returns.
            plot_idx (bool, optional): Whether to plot the index. Defaults to True.
            comparable_returns (list, optional): Returns of comparables (pd.Series). Defaults to [].
            rf (float, optional): Risk-free rate. Defaults to 0.0.
            zero_start (bool, optional): Whether return plots should start at 0. Defaults to False.
            save_as (str, optional): Path to save the image. Defaults to None.
            figsize (tuple): Figure size.
        """
        return_list = []
        metrics_list = []

        # Calculate returns and metrics for each selected portfolio
        for p in [p1 for p1 in self.portfolios if p1.name in portfolio_names]:
            metrics = p.compute_metrics(asset_ret=asset_returns, idx_ret=idx_returns, rf=rf)
            metrics_list.append(metrics)
            return_list.append(p.compute_returns(asset_returns))

        # Calculate metrics for the index if indicated.
        if plot_idx:
            metrics = p.compute_idx_metrics(idx_returns, rf)
            metrics_list.append(metrics)

        # Concatenate returns (including the index and other comparables if indicated)
        if plot_idx:
            df = pd.concat(return_list + [idx_returns] + comparable_returns, axis=1)
        else:
            df = pd.concat(return_list + comparable_returns, axis=1)

        # Calculate accumulated returns
        df = df.cumsum()

        if zero_start:
            df = df - df.iloc[0, :]

        # Create the figure for the accumulated returns plot
        fig, ax = plt.subplots(figsize=figsize)

        for col in df.columns:
            ax.plot(df.index, df[col], label=col)

        ax.set_title("Accumulated Returns Comparison", fontsize=14)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Accumulated Returns", fontsize=12)
        ax.legend(loc="upper left")

        plt.tight_layout()

        # Save the image if specified
        if save_as is not None:
            plt.savefig(save_as)

        plt.show()

        # Build and print the metrics table to the console
        if plot_idx:
            metrics_df = pd.DataFrame(metrics_list, index=portfolio_names + [idx_returns.columns[0]])
        else:
            metrics_df = pd.DataFrame(metrics_list, index=portfolio_names)
        print(metrics_df)

    def plot_weight_comparison(
        self,
        portfolio_names: list,
        plot_ucits_limits=True,
        skip_zero_weights=True,
        zero_threshold: float = 0.0,
        save_as: str = None,
        figsize: tuple = (8, 4),
    ):
        """
        Plots the weight comparison for a list of portfolios and displays the data table in the console.

        Args:
            portfolio_names (list): List of portfolio names to compare.
            plot_ucits_limits (bool): Indicates whether to show UCITS limit lines.
            skip_zero_weights (bool): Indicates whether to skip zero weights.
            zero_threshold (float): Threshold to set values to zero.
            save_as (str): Path to save the plot, if specified.
            figsize (tuple): Figure size.
        """
        # Filter selected portfolios and create a DataFrame of weights
        data = {
            portfolio.name: portfolio.asset_weights
            for portfolio in self.portfolios
            if portfolio.name in portfolio_names
        }

        # Create DataFrame with weights and asset names as index
        df = pd.DataFrame(data, index=self.asset_names)

        # Apply threshold to set values to zero
        df[df < zero_threshold] = 0.0

        # Filter rows where all weights are zero
        if skip_zero_weights:
            df = df[df.sum(axis=1) > 0]

        # Configure the figure for the bar plot
        fig, ax = plt.subplots(figsize=figsize)
        n = len(df.columns)
        width = 0.70 / n  # width of bars for each portfolio
        ind = np.arange(len(df.index))  # positions of bars on the x-axis

        for i, portfolio_name in enumerate(df.columns):
            ax.bar(ind + i * width, df[portfolio_name], width, label=portfolio_name)

        # Add UCITS limit lines if requested
        if plot_ucits_limits:
            ax.axhline(0.05, color="r", linestyle="--", label="UCITS Limit 5%")
            ax.axhline(0.10, color="r", linestyle="--", label="UCITS Limit 10%")

        ax.set_xticks(ind + (n - 1) * width / 2)
        ax.set_xticklabels(df.index, rotation=45, ha="right")
        ax.set_xlabel("Assets")
        ax.set_ylabel("Weight")
        ax.set_title("Portfolio Weights Comparison")
        ax.legend()

        plt.tight_layout()

        # Save the plot if a file is specified
        if save_as is not None:
            plt.savefig(save_as)

        # Show the plot
        plt.show()

        # Print the data table to the console
        # Configure display options
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", None)

        print(df.T)

        # Reset options to their default values
        pd.reset_option("display.max_rows")
        pd.reset_option("display.max_columns")
        pd.reset_option("display.width")
        pd.reset_option("display.max_colwidth")

    def plot_weight_comparison_interactive(self, portfolio_names: list):
        """
        Plots the weight comparison for a list of portfolios interactively.

        Args:
            portfolio_names (list): List of portfolio names to compare.
        """
        weights = [p.asset_weights for p in self.portfolios if p.name in portfolio_names]
        asset_names = self.asset_names

        # Create a figure
        fig = go.Figure()

        # Number of bars
        n = len(asset_names)
        ind = np.arange(n)  # positions of bars on the x-axis
        width = 0.70  # width of bars

        # Plot each series of bars
        for i, (portfolio_weights, portfolio_name) in enumerate(zip(weights, portfolio_names)):
            fig.add_trace(
                go.Bar(
                    x=asset_names,
                    y=portfolio_weights,
                    name=portfolio_name,
                    offsetgroup=i,
                )
            )

        # Reference lines
        fig.add_shape(
            type="line",
            x0=-0.5,
            y0=0.05,
            x1=n - 0.5,
            y1=0.05,
            line=dict(color="Red", dash="dash"),
        )
        fig.add_shape(
            type="line",
            x0=-0.5,
            y0=0.10,
            x1=n - 0.5,
            y1=0.10,
            line=dict(color="Red", dash="dash"),
        )

        # Labels and title
        fig.update_layout(
            xaxis=dict(tickmode="array", tickvals=ind, ticktext=asset_names),
            xaxis_title="Asset",
            yaxis_title="Weight",
            title="Optimized Portfolio Weights",
            barmode="group",
        )

        # Show the plot
        fig.show()
