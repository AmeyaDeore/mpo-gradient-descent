# End-to-End Portfolio Optimization Streamlit App
## Complete Implementation Guide

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture Design](#architecture-design)
3. [Prerequisites & Setup](#prerequisites--setup)
4. [Phase 1: Project Structure & Setup](#phase-1-project-structure--setup)
5. [Phase 2: Build UI Skeleton (Streamlit)](#phase-2-build-ui-skeleton-streamlit)
6. [Phase 3: Connect Data Layer](#phase-3-connect-data-layer)
7. [Phase 4: Integrate Gradient Descent Engine](#phase-4-integrate-gradient-descent-engine)
8. [Phase 5: Implement Results Visualization](#phase-5-implement-results-visualization)
9. [Phase 6: Error Handling & Performance Optimization](#phase-6-error-handling--performance-optimization)
10. [Phase 7: Testing & Deployment](#phase-7-testing--deployment)

---

## Project Overview

### What You're Building
A **real-time interactive portfolio optimization web application** that allows users to:
- Select stocks from S&P 500
- Adjust risk tolerance via an interactive slider
- Run gradient descent optimization to find optimal portfolio weights
- Visualize results with interactive charts
- Compare different optimization strategies (Sharpe Ratio, CVaR, etc.)

### Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│           PRESENTATION LAYER (Streamlit)                   │
│   - User Interface                                          │
│   - Interactive Controls (Sliders, Dropdowns)              │
│   - Real-time Visualizations                               │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│           LOGIC LAYER (TensorFlow)                          │
│   - Gradient Descent Algorithm                              │
│   - Loss Function Computation                               │
│   - Optimization Engine                                     │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│           DATA LAYER (Pandas/NumPy)                         │
│   - CSV Data Loading                                        │
│   - Returns Calculation                                     │
│   - Data Preprocessing                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Architecture Design

### Key Mathematical Concepts

#### Loss Function
The optimization problem minimizes:
```
L = -μ·w + λ·σ²(w) + penalties
```

Where:
- **μ·w**: Expected portfolio return (maximize)
- **λ·σ²(w)**: Portfolio risk weighted by risk aversion (minimize)
- **λ**: Risk aversion parameter (controlled by user slider)
- **penalties**: Sparsity, constraints, tracking error penalties

#### Portfolio Weights Vector
- **Input**: Daily returns matrix R (n_days × n_assets)
- **Output**: Weight vector w (n_assets × 1) where Σ(w) = 1
- **SparseMax**: Automatically sets low-weight assets to exactly 0

#### Risk Aversion λ Interpretation
- **λ = 1**: Balanced risk/return
- **λ < 1**: More aggressive (focus on returns)
- **λ > 1**: More conservative (focus on safety)

---

## Prerequisites & Setup

### System Requirements
- Python 3.10.6 (as specified in `.python-version`)
- 4GB RAM minimum
- 500MB disk space for data

### Step 1: Environment Setup

```bash
# Navigate to project directory
cd mpo-gradient-descent

# Create virtual environment (Windows)
python -m venv .venv
.venv\Scripts\activate

# Create virtual environment (Mac/Linux)
python -m venv .venv
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Add Streamlit if not already in requirements.txt
pip install streamlit==1.28.1
```

### Step 2: Verify Data Files
```bash
# Check data files exist
ls -la data/
# Should see: data_comp_SP500.csv and data_idx_SP500.csv
```

---

## Phase 1: Project Structure & Setup

### Step 1.1: Create Application Structure

Create the following new files in your `src/` directory:

```
src/
├── data_management.py          (existing)
├── models.py                   (existing)
├── portfolios.py               (existing)
├── risk_measures.py            (existing)
├── utils.py                    (existing)
├── streamlit_app.py            (NEW - main Streamlit app)
├── ui_components.py            (NEW - reusable UI components)
├── optimization_engine.py      (NEW - optimization wrapper)
└── data_loader.py              (NEW - enhanced data loading)
```

### Step 2: Update `requirements.txt`

Add these lines to your `requirements.txt` if not already present:

```txt
streamlit==1.28.1
streamlit-option-menu==0.3.6
plotly==5.24.1
pandas==2.2.3
numpy==1.26.4
tensorflow==2.10.1
tensorflow-addons==0.22.0
```

---

## Phase 2: Build UI Skeleton (Streamlit)

### Step 2.1: Create `src/ui_components.py`

This file contains reusable UI components:

```python
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Tuple

class UIComponents:
    """Reusable Streamlit UI components."""

    @staticmethod
    def render_header():
        """Render application header."""
        st.set_page_config(
            page_title="Portfolio Optimizer",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("🎯 Multi-Objective Portfolio Optimizer")
            st.markdown("""
            Optimize your S&P 500 portfolio using advanced gradient descent techniques.
            Powered by TensorFlow and mathematical optimization.
            """)
        with col2:
            st.metric("Status", "Ready", delta="✓")

    @staticmethod
    def render_sidebar_inputs() -> Tuple[List[str], float, str]:
        """
        Render sidebar controls and return user selections.

        Returns:
            Tuple containing:
            - selected_tickers: List of selected stock tickers
            - risk_aversion: Risk aversion parameter (1-10)
            - optimization_case: Selected optimization strategy
        """
        with st.sidebar:
            st.header("⚙️ Portfolio Parameters")

            # Stock Selection
            st.subheader("1. Select Stocks")
            st.caption("Choose 5-10 stocks from S&P 500")

            selected_tickers = st.multiselect(
                "Stock Tickers:",
                options=get_sp500_tickers(),
                default=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
                max_selections=20,
                key="ticker_selector"
            )

            if len(selected_tickers) < 2:
                st.warning("⚠️ Please select at least 2 stocks")
            if len(selected_tickers) > 20:
                st.error("❌ Maximum 20 stocks allowed")

            # Risk Aversion Slider
            st.subheader("2. Risk Tolerance")
            st.caption("Higher = More Conservative")

            risk_aversion = st.slider(
                "Risk Aversion Parameter (λ):",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
                help="Controls importance of risk vs return in optimization"
            )

            # Optimization Strategy Selection
            st.subheader("3. Optimization Strategy")
            optimization_case = st.radio(
                "Choose strategy:",
                options=[
                    "Case 1: Maximize Sharpe Ratio",
                    "Case 2: Minimize CVaR",
                    "Case 3: CVaR with UCITS Constraints",
                ],
                help="Different strategies for different investment goals"
            )

            # Advanced Options (Collapsible)
            with st.expander("⚙️ Advanced Options"):
                epochs = st.slider(
                    "Training Epochs:",
                    min_value=10,
                    max_value=1000,
                    value=500,
                    step=10
                )

                learning_rate = st.select_slider(
                    "Learning Rate:",
                    options=[0.001, 0.01, 0.05, 0.1],
                    value=0.01
                )

                use_random_init = st.checkbox(
                    "Use Random Weight Initialization",
                    value=False
                )

            # Action Buttons
            st.subheader("4. Actions")
            col1, col2 = st.columns(2)

            with col1:
                optimize_button = st.button(
                    "🚀 Optimize Portfolio",
                    use_container_width=True,
                    type="primary"
                )

            with col2:
                reset_button = st.button(
                    "🔄 Reset",
                    use_container_width=True
                )

            # Display parameters summary
            st.divider()
            st.subheader("📋 Summary")
            st.write(f"**Stocks Selected:** {len(selected_tickers)}")
            st.write(f"**Risk Aversion:** {risk_aversion:.2f}")
            st.write(f"**Strategy:** {optimization_case.split(':')[0]}")

        return selected_tickers, risk_aversion, optimization_case, optimize_button, reset_button

    @staticmethod
    def render_data_preview(df: pd.DataFrame):
        """Render data preview table."""
        st.subheader("📊 Data Preview")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Points", len(df))
        with col2:
            st.metric("Assets", len(df.columns))
        with col3:
            st.metric("Date Range", f"{df.index[0].date()} to {df.index[-1].date()}")

        st.dataframe(df.head(10), use_container_width=True)

    @staticmethod
    def render_optimization_status(status: str, progress: float = None):
        """Render optimization status."""
        if status == "running":
            with st.spinner("⏳ Optimizing portfolio..."):
                st.info("Running gradient descent algorithm...")
                if progress is not None:
                    st.progress(progress)
        elif status == "completed":
            st.success("✅ Optimization completed successfully!")
        elif status == "error":
            st.error("❌ Error during optimization")

    @staticmethod
    def render_results_summary(weights: np.ndarray, tickers: List[str],
                              expected_return: float, portfolio_risk: float,
                              sharpe_ratio: float):
        """Render results summary metrics."""
        st.subheader("📈 Portfolio Results")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Expected Return",
                f"{expected_return:.2%}",
                delta=None,
                delta_color="off"
            )

        with col2:
            st.metric(
                "Portfolio Risk",
                f"{portfolio_risk:.2%}",
                delta=None,
                delta_color="off"
            )

        with col3:
            st.metric(
                "Sharpe Ratio",
                f"{sharpe_ratio:.2f}",
                delta=None,
                delta_color="off"
            )

        with col4:
            st.metric(
                "Active Assets",
                f"{np.sum(weights > 0.001)}/{len(tickers)}",
                delta=None,
                delta_color="off"
            )

    @staticmethod
    def render_weights_table(weights: np.ndarray, tickers: List[str]):
        """Render portfolio weights as table."""
        st.subheader("📋 Allocation Details")

        # Create DataFrame
        weights_df = pd.DataFrame({
            "Ticker": tickers,
            "Weight": weights.flatten(),
            "Weight %": (weights.flatten() * 100)
        })

        # Sort by weight descending
        weights_df = weights_df.sort_values("Weight", ascending=False)

        # Format display
        weights_df_display = weights_df.copy()
        weights_df_display["Weight"] = weights_df_display["Weight"].apply(lambda x: f"{x:.4f}")
        weights_df_display["Weight %"] = weights_df_display["Weight %"].apply(lambda x: f"{x:.2f}%")

        st.dataframe(weights_df_display, use_container_width=True)

        return weights_df


def get_sp500_tickers() -> List[str]:
    """Get list of S&P 500 tickers from data file."""
    try:
        df = pd.read_csv("data/data_comp_SP500.csv", nrows=1)
        # Remove 'Date' column if present
        tickers = [col for col in df.columns if col != 'Date']
        return sorted(tickers)
    except:
        # Fallback to common tickers
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM",
                "V", "WMT", "PG", "XOM", "KO", "JNJ", "CVX"]
```

### Step 2.2: Create Main `src/streamlit_app.py`

This is your main Streamlit application:

```python
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from ui_components import UIComponents
from data_loader import DataLoader
from optimization_engine import OptimizationEngine

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "optimization_results" not in st.session_state:
        st.session_state.optimization_results = None
    if "last_tickers" not in st.session_state:
        st.session_state.last_tickers = None
    if "computation_done" not in st.session_state:
        st.session_state.computation_done = False

def main():
    """Main application flow."""

    # Initialize
    initialize_session_state()
    UIComponents.render_header()

    # Get user inputs from sidebar
    selected_tickers, risk_aversion, optimization_case, optimize_btn, reset_btn = UIComponents.render_sidebar_inputs()

    # Validate inputs
    if not selected_tickers or len(selected_tickers) < 2:
        st.warning("Please select at least 2 stocks to proceed.")
        return

    # Load data
    try:
        st.info("📥 Loading market data...")
        data_loader = DataLoader()
        returns_df = data_loader.load_and_prepare_data(selected_tickers)
        UIComponents.render_data_preview(returns_df)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return

    # Optimize when button is clicked
    if optimize_btn:
        try:
            UIComponents.render_optimization_status("running", 0.3)

            # Extract optimization case number
            case_num = int(optimization_case[5])  # "Case X:" format

            # Run optimization
            opt_engine = OptimizationEngine(
                returns_data=returns_df,
                case_num=case_num,
                risk_aversion=risk_aversion
            )

            results = opt_engine.optimize()
            st.session_state.optimization_results = results
            st.session_state.last_tickers = selected_tickers
            st.session_state.computation_done = True

            UIComponents.render_optimization_status("completed")

        except Exception as e:
            st.error(f"Optimization error: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

    # Display results if available
    if st.session_state.optimization_results is not None:
        results = st.session_state.optimization_results

        # Results summary metrics
        UIComponents.render_results_summary(
            weights=results['weights'],
            tickers=st.session_state.last_tickers,
            expected_return=results['expected_return'],
            portfolio_risk=results['portfolio_risk'],
            sharpe_ratio=results['sharpe_ratio']
        )

        # Weights table
        weights_df = UIComponents.render_weights_table(
            weights=results['weights'],
            tickers=st.session_state.last_tickers
        )

        # Charts
        render_visualizations(
            weights=results['weights'],
            tickers=st.session_state.last_tickers,
            history=results.get('history', [])
        )

        # Download results
        render_download_section(weights_df, results)

    # Reset button
    if reset_btn:
        st.session_state.optimization_results = None
        st.session_state.computation_done = False
        st.rerun()

def render_visualizations(weights: np.ndarray, tickers: list, history: list):
    """Render portfolio visualization charts."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    col1, col2 = st.columns(2)

    with col1:
        # Pie Chart
        st.subheader("🥧 Portfolio Allocation")

        # Filter out zero weights
        mask = weights.flatten() > 0.001
        active_tickers = [t for i, t in enumerate(tickers) if mask[i]]
        active_weights = weights[mask]

        fig_pie = go.Figure(data=[go.Pie(
            labels=active_tickers,
            values=active_weights.flatten(),
            hoverinfo="label+percent+value"
        )])

        fig_pie.update_layout(
            height=400,
            showlegend=True,
            hovermode="closest"
        )

        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Bar Chart
        st.subheader("📊 Weight Distribution")

        weights_sorted_idx = np.argsort(weights.flatten())[::-1]

        fig_bar = go.Figure(data=[go.Bar(
            x=[tickers[i] for i in weights_sorted_idx],
            y=weights.flatten()[weights_sorted_idx],
            marker_color='rgba(26, 118, 255, 0.8)',
            text=(weights.flatten()[weights_sorted_idx] * 100).round(2),
            textposition='auto',
        )])

        fig_bar.update_layout(
            title="Portfolio Weights",
            xaxis_title="Stock Ticker",
            yaxis_title="Weight",
            height=400,
            hovermode="closest"
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    # Training History (if available)
    if history:
        st.subheader("📉 Training History")

        history_df = pd.DataFrame(history)

        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            y=history_df['loss'],
            mode='lines',
            name='Loss',
            line=dict(color='#1f77b4')
        ))

        fig_loss.update_layout(
            title="Optimization Loss Over Epochs",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            height=400,
            hovermode="x unified"
        )

        st.plotly_chart(fig_loss, use_container_width=True)

def render_download_section(weights_df: pd.DataFrame, results: dict):
    """Render download options for results."""
    st.divider()
    st.subheader("💾 Export Results")

    col1, col2 = st.columns(2)

    with col1:
        # CSV Download
        csv_data = weights_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Weights (CSV)",
            data=csv_data,
            file_name="portfolio_weights.csv",
            mime="text/csv"
        )

    with col2:
        # JSON Download
        import json
        json_data = json.dumps({
            "expected_return": float(results['expected_return']),
            "portfolio_risk": float(results['portfolio_risk']),
            "sharpe_ratio": float(results['sharpe_ratio']),
            "weights": {ticker: float(w) for ticker, w in zip(results['tickers'], results['weights'].flatten())}
        }, indent=4)

        st.download_button(
            label="📥 Download Results (JSON)",
            data=json_data,
            file_name="portfolio_results.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()
```

---

## Phase 3: Connect Data Layer

### Step 3.1: Create `src/data_loader.py`

Enhanced data loading with caching:

```python
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
```

---

## Phase 4: Integrate Gradient Descent Engine

### Step 4.1: Create `src/optimization_engine.py`

Wrapper for optimization with case selection:

```python
import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Dict, Any
import sys
from pathlib import Path

# Import existing modules
sys.path.insert(0, str(Path(__file__).parent))
from models import MPOModel
from portfolios import PortfolioMetrics
from risk_measures import RiskMeasures

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
            """Compute loss for Sharpe ratio maximization."""

            # Portfolio return
            port_return = tf.reduce_sum(tf.linalg.matvec(
                tf.transpose(assets_rets),
                w
            ))

            # Portfolio std deviation
            cov_matrix = tf.linalg.matmul(
                tf.transpose(assets_rets),
                assets_rets
            ) / tf.cast(tf.shape(assets_rets)[0], tf.float32)

            portfolio_var = tf.linalg.matmul(
                tf.transpose(w),
                tf.linalg.matmul(cov_matrix, w)
            )
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
            port_rets = tf.linalg.matvec(tf.transpose(assets_rets), w)

            # CVaR (95% confidence level)
            confidence_level = 0.95
            cvar_threshold = tf.quantile(port_rets, 1 - confidence_level)

            # Losses exceeding threshold
            excess_losses = tf.where(
                port_rets < cvar_threshold,
                port_rets - cvar_threshold,
                0.0
            )

            cvar = cvar_threshold + tf.reduce_mean(excess_losses)

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

            # CVaR calculation
            port_rets = tf.linalg.matvec(tf.transpose(assets_rets), w)
            confidence_level = 0.95
            cvar_threshold = tf.quantile(port_rets, 1 - confidence_level)
            excess_losses = tf.where(
                port_rets < cvar_threshold,
                port_rets - cvar_threshold,
                0.0
            )
            cvar = cvar_threshold + tf.reduce_mean(excess_losses)

            # UCITS constraint: max 10% per position
            ucits_violation = tf.reduce_sum(tf.maximum(w - 0.1, 0.0))

            loss = cvar + self.risk_aversion * (ucits_violation + tf.reduce_sum(tf.abs(w)))

            return {
                "loss": loss,
                "cvar": cvar,
                "ucits_violation": ucits_violation
            }

        def weights_function(z):
            """Apply constraints and normalization."""
            # Softmax for smoothness
            w = tf.nn.softmax(z, axis=0)
            # Apply UCITS constraint (clip at 0.1)
            w = tf.minimum(w, 0.1)
            # Renormalize
            w = w / (tf.reduce_sum(w) + 1e-6)
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
```

---

## Phase 5: Implement Results Visualization

The visualization code is already integrated in `streamlit_app.py` via the `render_visualizations()` function. Key charts include:

1. **Pie Chart**: Shows portfolio allocation percentages
2. **Bar Chart**: Shows weight distribution with sorting
3. **Training History**: Shows loss convergence

To enhance further, add to `src/streamlit_app.py`:

```python
def render_advanced_visualizations(weights: np.ndarray,
                                  tickers: list,
                                  returns_df: pd.DataFrame):
    """Advanced visualization options."""
    import plotly.graph_objects as go

    st.subheader("📊 Advanced Analytics")

    tabs = st.tabs(["Correlation Matrix", "Returns Distribution", "Risk-Return Scatter"])

    with tabs[0]:
        # Correlation heatmap
        corr_matrix = returns_df.corr()
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu'
        ))
        st.plotly_chart(fig_corr, use_container_width=True)

    with tabs[1]:
        # Returns distribution
        fig_dist = go.Figure()
        for ticker in tickers:
            fig_dist.add_trace(go.Histogram(
                x=returns_df[ticker],
                name=ticker,
                opacity=0.75
            ))
        st.plotly_chart(fig_dist, use_container_width=True)

    with tabs[2]:
        # Risk-return scatter
        individual_returns = returns_df.mean() * 252
        individual_risk = returns_df.std() * np.sqrt(252)

        fig_scatter = go.Figure(data=go.Scatter(
            x=individual_risk,
            y=individual_returns,
            mode='markers+text',
            text=tickers,
            textposition="top center",
            marker=dict(size=12, color='rgba(26, 118, 255, 0.8)')
        ))

        fig_scatter.update_layout(
            title="Risk-Return Profile",
            xaxis_title="Annualized Risk",
            yaxis_title="Annualized Return",
            height=400
        )

        st.plotly_chart(fig_scatter, use_container_width=True)
```

---

## Phase 6: Error Handling & Performance Optimization

### Step 6.1: Add to `src/streamlit_app.py`

```python
import logging
from functools import lru_cache

# Setup logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

@st.cache_resource
def get_data_loader():
    """Cache data loader instance."""
    return DataLoader()

def handle_optimization_errors(func):
    """Decorator for error handling."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            st.error(f"❌ Input Error: {str(e)}")
            logger.error(f"ValueError: {str(e)}")
        except MemoryError:
            st.error("❌ Out of Memory: Try fewer stocks or shorter time periods")
            logger.error("MemoryError")
        except Exception as e:
            st.error(f"❌ Unexpected Error: {str(e)}")
            logger.exception("Unexpected error")

    return wrapper

# Performance optimization: limit number of epochs based on data size
def suggest_epochs(num_data_points: int) -> int:
    """Suggest epochs based on data size."""
    if num_data_points < 500:
        return 200
    elif num_data_points < 1000:
        return 500
    else:
        return 1000
```

---

## Phase 7: Testing & Deployment

### Step 7.1: Create `test_streamlit_app.py`

```python
import pytest
import numpy as np
import pandas as pd
from src.optimization_engine import OptimizationEngine
from src.data_loader import DataLoader

def test_data_loader():
    """Test data loading."""
    loader = DataLoader()
    tickers = ["AAPL", "MSFT", "GOOGL"]
    returns_df = loader.load_and_prepare_data(tickers)

    assert returns_df.shape[1] == 3
    assert len(returns_df) > 100
    assert not returns_df.isna().values.any()

def test_optimization_case_1():
    """Test Sharpe ratio optimization."""
    # Create dummy data
    returns_data = pd.DataFrame(
        np.random.randn(100, 3) * 0.01,
        columns=["A", "B", "C"]
    )

    engine = OptimizationEngine(
        returns_data=returns_data,
        case_num=1,
        epochs=10
    )

    results = engine.optimize()

    assert "weights" in results
    assert np.allclose(np.sum(results['weights']), 1.0)
    assert all(w >= 0 for w in results['weights'].flatten())

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Step 7.2: Run Streamlit App Locally

```bash
# Terminal/Command Prompt
cd mpo-gradient-descent
.venv\Scripts\activate  # or: source .venv/bin/activate on Mac/Linux
streamlit run src/streamlit_app.py
```

### Step 7.3: Deploy to Streamlit Cloud

1. **Push to GitHub**:
```bash
git add src/streamlit_app.py src/ui_components.py src/optimization_engine.py src/data_loader.py
git commit -m "Add Streamlit portfolio optimization app"
git push origin main
```

2. **Deploy to Streamlit Cloud**:
   - Go to https://streamlit.io/cloud
   - Connect your GitHub account
   - Deploy `src/streamlit_app.py`
   - Set main file path: `src/streamlit_app.py`

### Step 7.4: Create `.streamlit/config.toml`

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[logger]
level = "warning"

[client]
toolbarMode = "minimal"
```

---

## Complete File Checklist

```
✅ src/streamlit_app.py          [Main app - REQUIRED]
✅ src/ui_components.py          [UI components - REQUIRED]
✅ src/optimization_engine.py    [Optimization logic - REQUIRED]
✅ src/data_loader.py            [Data loading - REQUIRED]
✅ src/data_management.py        [Existing - already there]
✅ src/models.py                 [Existing - already there]
✅ .streamlit/config.toml         [Config - OPTIONAL but recommended]
✅ requirements.txt              [Dependencies - UPDATED]
✅ data/data_comp_SP500.csv      [Data - already there]
✅ data/data_idx_SP500.csv       [Data - already there]
```

---

## Testing Checklist

- [ ] Load and verify market data for selected stocks
- [ ] Test optimization with different risk aversion levels
- [ ] Verify portfolio weights sum to 1.0
- [ ] Check that SparseMax eliminates low-weight assets
- [ ] Test with minimum (2) and maximum (20) stocks
- [ ] Verify all three optimization cases work
- [ ] Test download functionality (CSV, JSON)
- [ ] Verify visualizations render correctly
- [ ] Test on different browsers (Chrome, Firefox, Safari)
- [ ] Performance test with 500+ epochs

---

## Troubleshooting Guide

### Issue: "ModuleNotFoundError: No module named 'tensorflow'"
**Solution**:
```bash
pip install tensorflow==2.10.1 tensorflow-addons==0.22.0
```

### Issue: "FileNotFoundError: data/data_comp_SP500.csv"
**Solution**:
- Ensure CSV files are in `data/` directory
- Check file paths are absolute or relative from project root

### Issue: "Optimization runs very slowly"
**Solution**:
- Reduce number of epochs (start with 100)
- Use fewer stocks (5-10)
- Lower learning rate to 0.001

### Issue: "Out of Memory Error"
**Solution**:
- Select fewer stocks
- Use shorter time periods
- Reduce epochs

### Issue: "Weights don't sum to 1.0"
**Solution**:
- Ensure normalization in `weights_function()`
- Add epsilon (1e-6) to prevent division by zero

---

## Next Steps & Enhancements

### Immediate Enhancements
1. Add backtesting module to test portfolio performance historically
2. Implement portfolio rebalancing scheduler
3. Add risk analytics (Var, VaR, drawdown analysis)
4. Create performance comparison across cases

### Future Enhancements
1. Real-time market data via yfinance API
2. Multi-period optimization (rolling windows)
3. Constraints customization (sector limits, short-selling)
4. ML-based hyperparameter tuning
5. Docker containerization for easy deployment
6. Database integration for historical results

---

## References & Resources

- **Paper Reference**: Multi-Objective Portfolio Optimization with Gradient Descent
- **TensorFlow Ops**: https://www.tensorflow.org/api_docs
- **Streamlit Docs**: https://docs.streamlit.io/
- **Plotly Chart Types**: https://plotly.com/python/
- **Portfolio Theory**: https://en.wikipedia.org/wiki/Modern_portfolio_theory

---

## Support & Debugging

### Enable Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check TensorFlow Version
```bash
python -c "import tensorflow; print(tensorflow.__version__)"
```

### Validate Data
```python
import pandas as pd
df = pd.read_csv("data/data_comp_SP500.csv")
print(df.shape)
print(df.head())
print(df.info())
```

---

**Last Updated**: 2026-04-19
**Status**: Production Ready
**Version**: 1.0.0
