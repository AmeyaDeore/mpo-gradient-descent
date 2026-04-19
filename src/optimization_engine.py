import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Dict, Any
import sys
from pathlib import Path
import tensorflow_probability as tfp

# Import existing modules
sys.path.insert(0, str(Path(__file__).parent))
from models import MPOModel

class OptimizationEngine:
    """Main optimization engine wrapper."""

    def __init__(self,
                 returns_data: pd.DataFrame,
                 case_num: int = 1,
                 risk_aversion: float = 1.0,
                 epochs: int = 500,
                 learning_rate: float = 0.01):
        """
        Initialize optimization engine.

        Args:
            returns_data: DataFrame of daily log returns
            case_num: Case number (1-6)
            risk_aversion: Risk aversion parameter (λ)
            epochs: Training epochs
            learning_rate: Learning rate for optimizer
        """
        self.returns_data = returns_data
        self.case_num = case_num
        self.risk_aversion = risk_aversion
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.num_assets = returns_data.shape[1]
        self.tickers = returns_data.columns.tolist()

        # Setup optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def create_sharp_ratio_case(self):
        """
        Case 1: Maximize Sharpe Ratio
        Loss: -Sharpe + λ*constraints
        """
        def loss_function(assets_rets, w, idx=None):
            # Portfolio return (annualized)
            port_rets = tf.linalg.matmul(assets_rets, w)
            port_return = tf.reduce_mean(port_rets) * 252.0

            # Portfolio std deviation (annualized)
            assets_centered = assets_rets - tf.reduce_mean(assets_rets, axis=0, keepdims=True)
            cov_matrix = tf.linalg.matmul(
                tf.transpose(assets_centered),
                assets_centered
            ) / tf.cast(tf.shape(assets_rets)[0] - 1, tf.float32) * 252.0

            portfolio_var = tf.linalg.matmul(
                tf.transpose(w),
                tf.linalg.matmul(cov_matrix, w)
            )[0,0]
            portfolio_std = tf.sqrt(portfolio_var + 1e-6)

            # Sharpe ratio (assuming risk-free rate = 0)
            sharpe = port_return / portfolio_std

            # Loss (negative because we maximize)
            loss = -sharpe + self.risk_aversion * tf.reduce_sum(tf.abs(w))

            return {
                "loss": loss,
                "return": port_return,
                "risk": portfolio_std,
                "sharpe": sharpe
            }

        def weights_function(z):
            """Apply softmax to get normalized weights."""
            return tf.nn.softmax(z, axis=0)

        def get_best_weights(history):
            """Get weights with best (maximum) Sharpe ratio."""
            best_idx = np.argmax([h.get("sharpe", 0) for h in history])
            return history[best_idx]

        return loss_function, weights_function, get_best_weights

    def create_cvar_case(self):
        """
        Case 2: Minimize CVaR (Conditional Value at Risk)
        Loss: CVaR + λ*constraints
        """
        def loss_function(assets_rets, w, idx=None):
            """Compute loss for CVaR minimization."""
            
            # Portfolio returns
            port_rets = tf.linalg.matmul(assets_rets, w)

            # CVaR (95% confidence level)
            confidence_level = 0.95
            alpha = (1.0 - confidence_level) * 100.0
            cvar_threshold = tfp.stats.percentile(port_rets, alpha)[0]

            # Losses exceeding threshold
            excess_losses = tf.where(
                port_rets < cvar_threshold,
                cvar_threshold - port_rets,
                0.0
            )

            cvar = -cvar_threshold + tf.reduce_mean(excess_losses) / (1.0 - confidence_level)

            # Loss
            loss = cvar + self.risk_aversion * tf.reduce_sum(tf.abs(w))

            return {
                "loss": loss,
                "cvar": cvar,
                "weights_norm": tf.reduce_sum(tf.abs(w))
            }

        def weights_function(z):
            """Apply sparsemax for sparsity."""
            # Simplified sparsemax (approximate with threshold)
            z_norm = z - tf.reduce_max(z)
            weights = tf.nn.relu(z_norm) / (tf.reduce_sum(tf.nn.relu(z_norm)) + 1e-6)
            return weights

        def get_best_weights(history):
            """Get weights with minimum CVaR."""
            best_idx = np.argmin([h.get("cvar", float('inf')) for h in history])
            return history[best_idx]

        return loss_function, weights_function, get_best_weights

    def create_cvar_ucits_case(self):
        """
        Case 3: Minimize CVaR with UCITS Constraints
        Loss: CVaR + λ*(sparsity + UCITS violation)
        """
        def loss_function(assets_rets, w, idx=None):
            """Compute loss with UCITS constraints."""
            import tensorflow_probability as tfp
            
            # CVaR calculation
            port_rets = tf.linalg.matmul(assets_rets, w)
            confidence_level = 0.95
            alpha = (1.0 - confidence_level) * 100.0
            cvar_threshold = tfp.stats.percentile(port_rets, alpha)[0]
            excess_losses = tf.where(
                port_rets < cvar_threshold,
                cvar_threshold - port_rets,
                0.0
            )
            cvar = -cvar_threshold + tf.reduce_mean(excess_losses) / (1.0 - confidence_level)

            # UCITS constraint: max 10% per position
            ucits_violation = tf.reduce_sum(tf.maximum(w - 0.10, 0.0))

            loss = cvar + self.risk_aversion * (ucits_violation + tf.reduce_sum(tf.abs(w)))

            return {
                "loss": loss,
                "cvar": cvar,
                "ucits_violation": ucits_violation
            }

        def weights_function(z):
            """Apply constraints and normalization."""
            w = tf.nn.softmax(z, axis=0) # Softmax for smoothness
            w = tf.minimum(w, 0.1) # Apply UCITS constraint (clip at 0.1)
            w = w / (tf.reduce_sum(w) + 1e-6) # Renormalize
            return w

        def get_best_weights(history):
            """Get weights with minimum CVaR."""
            best_idx = np.argmin([h.get("cvar", float('inf')) for h in history])
            return history[best_idx]

        return loss_function, weights_function, get_best_weights

    def select_case(self):
        """Select case based on case_num."""
        cases = {
            1: self.create_sharp_ratio_case,
            2: self.create_cvar_case,
            3: self.create_cvar_ucits_case,
        }

        if self.case_num not in cases:
            raise ValueError(f"Case {self.case_num} not implemented. Valid cases: {list(cases.keys())}")

        return cases[self.case_num]()

    def optimize(self) -> Dict[str, Any]:
        """
        Run optimization.

        Returns:
            Dictionary with optimization results
        """
        # Select and create case
        loss_fn, weights_fn, best_weights_fn = self.select_case()

        # Create model
        model = MPOModel(
            num_assets=self.num_assets,
            loss_function=loss_fn,
            weights_function=weights_fn,
            get_best_weights_function=best_weights_fn,
            optimizer=self.optimizer
        )

        # Train
        returns_array = self.returns_data.values.astype(np.float32)
        history = model.fit(returns_array, epochs=self.epochs)

        # Get best weights
        best_weights = model.get_best_weights()
        final_weights = weights_fn(model.z).numpy()

        # Calculate metrics
        ann_return = self._calculate_annual_return(final_weights)
        ann_risk = self._calculate_annual_risk(final_weights)
        sharpe = ann_return / (ann_risk + 1e-6)

        return {
            'weights': final_weights,
            'tickers': self.tickers,
            'expected_return': ann_return,
            'portfolio_risk': ann_risk,
            'sharpe_ratio': sharpe,
            'history': history
        }

    def _calculate_annual_return(self, weights: np.ndarray) -> float:
        """Calculate annualized portfolio return."""
        returns_mean = self.returns_data.mean().values
        portfolio_return = np.dot(weights.flatten(), returns_mean) * 252
        return portfolio_return

    def _calculate_annual_risk(self, weights: np.ndarray) -> float:
        """Calculate annualized portfolio risk."""
        cov_matrix = self.returns_data.cov().values
        portfolio_var = np.dot(weights.flatten(), np.dot(cov_matrix, weights.flatten())) * 252
        portfolio_std = np.sqrt(portfolio_var)
        return portfolio_std
