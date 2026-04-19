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

        # Custom CSS for premium aesthetics
        st.markdown("""
        <style>
        /* Glassmorphism for sidebar */
        [data-testid="stSidebar"] {
            background-color: rgba(20, 28, 43, 0.6) !important;
            backdrop-filter: blur(10px) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        /* Buttons glow effect */
        .stButton>button {
            border-radius: 8px !important;
            background: linear-gradient(90deg, #00E5FF, #2979FF) !important;
            color: white !important;
            font-weight: 600 !important;
            border: none !important;
            transition: all 0.3s ease !important;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0, 229, 255, 0.4);
        }
        /* Metric cards glow */
        [data-testid="stMetricValue"] {
            font-size: 2.2rem !important;
            font-weight: 700 !important;
            background: -webkit-linear-gradient(45deg, #00E5FF, #FFFFFF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        /* Standard metric containers */
        [data-testid="metric-container"] {
            background-color: rgba(20, 28, 43, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        </style>
        """, unsafe_allow_html=True)

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
    def render_sidebar_inputs() -> Tuple[List[str], float, str, bool, bool]:
        """
        Render sidebar controls and return user selections.

        Returns:
            Tuple containing:
            - selected_tickers: List of selected stock tickers
            - risk_aversion: Risk aversion parameter (1-10)
            - optimization_case: Selected optimization strategy
            - optimize_button: Boolean, True if clicked
            - reset_button: Boolean, True if clicked
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
