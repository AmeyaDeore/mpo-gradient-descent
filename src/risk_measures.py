import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from riskfolio import RiskFunctions as rpf


class RiskMeasures:
    def Sharpe_Ratio(X: np.ndarray, r_f: float = 0.0) -> float:
        r"""
        Calculate the Sharpe Ratio.

        .. math::
            RS = \frac{r_p-r_f}{\sigma(r_p)}

        Parameters
        ----------
        X : 1d-array
            Returns series, must have Tx1 size.
            Portfolio returns.
        r_f : float
            Risk free asset return.

        Raises
        ------
        ValueError
            When the value cannot be calculated.

        Returns
        -------
        value : float
            Sharpe Ratio of a returns series.
        """

        a = np.array(X, ndmin=2)
        if a.shape[0] == 1 and a.shape[1] > 1:
            a = a.T
        if a.shape[0] > 1 and a.shape[1] > 1:
            raise ValueError("returns must have Tx1 size")

        value = rpf.Sharpe(returns=a, rf=r_f)
        value = np.float32(value)

        return value

    def Std(X: np.ndarray) -> float:
        r"""
        Calculate the Standar Deviation.

        .. math::
            std = \sigma(r_p)

        Parameters
        ----------
        X : 1d-array
            Returns series, must have Tx1 size.
            Portfolio returns.

        Raises
        ------
        ValueError
            When the value cannot be calculated.

        Returns
        -------
        value : float
            Standar Deviation of a returns series.
        """

        a = np.array(X, ndmin=2)
        if a.shape[0] == 1 and a.shape[1] > 1:
            a = a.T
        if a.shape[0] > 1 and a.shape[1] > 1:
            raise ValueError("returns must have Tx1 size")

        value = np.std(a)
        value = np.float32(value)

        return value

    def Tracking_Error(X: np.ndarray, Y: np.ndarray) -> float:
        r"""
        Calculate the Tracking Error of a returns series.

        .. math::
            TE = \sqrt{\sum_{t=1}^{T}(Y_{t}-\hat{Y}_{t})^{2}/(T-1)}

        Parameters
        ----------
        X : 1d-array
            Returns series, must have Tx1 size.
            Marpfet index returns.
        Y : 1d-array
            Returns series, must have Tx1 size.
            Asset returns.

        Raises
        ------
        ValueError
            When the value cannot be calculated.

        Returns
        -------
        value : float
            Tracking Error of a returns series.
        """

        a = np.array(X, ndmin=2)
        b = np.array(Y, ndmin=2)
        if a.shape[0] == 1 and a.shape[1] > 1:
            a = a.T
        if b.shape[0] == 1 and b.shape[1] > 1:
            b = b.T
        if a.shape[0] > 1 and a.shape[1] > 1:
            raise ValueError("returns must have Tx1 size")
        if b.shape[0] > 1 and b.shape[1] > 1:
            raise ValueError("returns must have Tx1 size")

        # The tracking error is the standard deviation of the difference between the Portfolio
        # and benchmarpf returns.
        # T = len(b)
        # value = np.sqrt(np.sum(b - a) ** 2 / (T - 1))
        value = np.std(a - b)
        value = np.float32(value)

        return value

    def VaR_Hist(X: np.ndarray, alpha: float = 0.05) -> float:
        r"""
        Calculate the Value at Risk (VaR) of a returns series.

        .. math::
            \text{VaR}_{\alpha}(X) = -\inf_{t \in (0,T)} \left \{ X_{t} \in
            \mathbb{R}: F_{X}(X_{t})>\alpha \right \}

        Parameters
        ----------
        X : 1d-array
            Returns series, must have Tx1 size.
        alpha : float, optional
            Significance level of VaR. The default is 0.05.
        Raises
        ------
        ValueError
            When the value cannot be calculated.

        Returns
        -------
        value : float
            VaR of a returns series.
        """

        value = rpf.VaR_Hist(X, alpha=alpha)
        value = np.float32(value)

        return value

    def CVaR_Hist(X: np.ndarray, alpha: float = 0.05) -> float:
        r"""
        Calculate the Conditional Value at Risk (CVaR) of a returns series.

        .. math::
            \text{CVaR}_{\alpha}(X) = \text{VaR}_{\alpha}(X) +
            \frac{1}{\alpha T} \sum_{t=1}^{T} \max(-X_{t} -
            \text{VaR}_{\alpha}(X), 0)

        Parameters
        ----------
        X : 1d-array
            Returns series, must have Tx1 size.
        alpha : float, optional
            Significance level of CVaR. The default is 0.05.

        Raises
        ------
        ValueError
            When the value cannot be calculated.

        Returns
        -------
        value : float
            CVaR of a returns series.
        """

        value = rpf.CVaR_Hist(X, alpha=alpha)
        value = np.float32(value)

        return value

    def Max_Drawdown(X: np.ndarray) -> float:
        r"""
        Calculate the Maximum Drawdown of a prices series.

        .. math::
            MDD = \max_{t \in (0,T)} \left \{ \frac{X_{t}-\max_{t \in (0,T)}X_{t}}{\max_{t \in (0,T)}X_{t}} \right \}

        Parameters
        ----------
        X : 1d-array
            Prices series, must have Tx1 size.

        Raises
        ------
        ValueError
            When the value cannot be calculated.

        Returns
        -------
        value : float
            Maximum Drawdown of a returns series.
        """

        a = np.array(X, ndmin=2)
        if a.shape[0] == 1 and a.shape[1] > 1:
            a = a.T
        if a.shape[0] > 1 and a.shape[1] > 1:
            raise ValueError("returns must have Tx1 size")

        a = pd.Series(a.flatten())
        value = np.max(np.abs((a - a.cummax()) / a.cummax()))
        value = np.float32(value)

        return value

    def calculate_metrics(w: np.ndarray, asset_ret: np.ndarray, idx_ret: np.ndarray, r_f: float = 0.0) -> tuple:
        """Calculate portfolio metrics and portfolio metrics relative to the index.

        Args:
            w (np.ndarray): weights of the portfolio assets
            asset_ret (np.ndarray): returns of the assets
            idx_ret (np.ndarray): returns of the index
            r_f (float): Risk-free asset return. Defaults to 0.0

        Returns:
            tuple: sharpe, tracking_error, var, cvar, weights greater than 5%
        """
        p_ret = np.dot(asset_ret, w)

        s = RiskMeasures.Sharpe_Ratio(p_ret, r_f)
        te = RiskMeasures.Tracking_Error(idx_ret, p_ret)
        var = RiskMeasures.VaR_Hist(p_ret, alpha=0.05)
        cvar = RiskMeasures.CVaR_Hist(p_ret, alpha=0.05)
        mdd = RiskMeasures.Max_Drawdown(p_ret.cumsum())

        # Weights exceeding 5%
        mask = np.where(w > 0.05, 1, 0)
        weights_005 = np.sum(w * mask)

        return s, te, var, cvar, weights_005, mdd

    def Sharpe_Ratio_tf(x: tf.Tensor, r_f: tf.Tensor) -> tf.Tensor:
        r"""
        Calculate the Sharpe Ratio.

        Parameters
        ----------
        x : tf.Tensor
            Returns series, must have Tx1 size.
            Portfolio returns.
        r_f : tf.Tensor
            Risk free asset return.

        Returns
        -------
        value : tf.Tensor
            Sharpe Ratio of a returns series.
        """
        value = (tf.math.reduce_mean(x) - r_f) / tf.math.reduce_std(x)

        return value

    def Std_tf(x: tf.Tensor) -> tf.Tensor:
        r"""
        Calculate the Standard Deviation.

        Parameters
        ----------
        x : tf.Tensor
            Returns series, must have Tx1 size.
            Portfolio returns.

        Returns
        -------
        value : tf.Tensor
            Standard Deviation of a returns series.
        """
        value = tf.math.reduce_std(x)

        return value

    def VaR_tf(x: tf.Tensor, alpha: float = 0.05) -> tf.Tensor:
        """Calculates VaR at the specified confidence level.

        Args:
            x (tf.Tensor): Series of returns.
            alpha (float, optional): Percentile. Defaults to 0.05.

        Returns:
            float: Value at Risk (VaR)
        """
        # Calculate Value at Risk (VaR)
        var = -tfp.stats.percentile(x, q=alpha * 100)
        return var

    def CVaR_tf(x: tf.Tensor, alpha: float = 0.05) -> tf.Tensor:
        """Calculates CVaR at the specified confidence level.

        Args:
            x (tf.Tensor): Series of returns.
            alpha (float, optional): Percentile. Defaults to 0.05.

        Returns:
            float: Conditional Value at Risk (CVaR)
        """
        # Calculate Value at Risk (VaR) and make it positive.
        var_alpha = RiskMeasures.VaR_tf(x, alpha=alpha)

        # Calculate the loss exceeding VaR.
        shortfall = tf.nn.relu(-x - var_alpha)

        # Sum and scale.
        mean_shortfall = tf.math.reduce_mean(shortfall) / alpha

        # CVaR = VaR + mean_shortfall
        cvar = var_alpha + mean_shortfall
        return cvar

    def TrackingError_tf(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """Calculate the Tracking Error between two return series.

        Args:
            x (tf.Tensor): Series of returns.
            y (tf.Tensor): Series of returns.

        Returns:
            float: Tracking Error
        """
        # te = tf.math.reduce_std(x - y)
        te = tfp.stats.stddev(x - y)
        return te
