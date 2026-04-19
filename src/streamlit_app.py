import streamlit as st
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from ui_components import UIComponents
from data_loader import DataLoader
from optimization_engine import OptimizationEngine

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
            st.error(traceback.format_exc())
    return wrapper

def suggest_epochs(num_data_points: int) -> int:
    """Suggest epochs based on data size."""
    if num_data_points < 500:
        return 200
    elif num_data_points < 1000:
        return 500
    else:
        return 1000

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "optimization_results" not in st.session_state:
        st.session_state.optimization_results = None
    if "last_tickers" not in st.session_state:
        st.session_state.last_tickers = None
    if "computation_done" not in st.session_state:
        st.session_state.computation_done = False

@handle_optimization_errors
def perform_optimization(selected_tickers, risk_aversion, optimization_case, returns_df):
    UIComponents.render_optimization_status("running", 0.3)
    
    # Extract optimization case number
    case_num = int(optimization_case[5])  # "Case X:" format

    # Determine epochs
    num_data_points = len(returns_df) * len(selected_tickers)
    epochs = suggest_epochs(num_data_points)
    
    # Run optimization
    opt_engine = OptimizationEngine(
        returns_data=returns_df,
        case_num=case_num,
        risk_aversion=risk_aversion,
        epochs=epochs
    )

    results = opt_engine.optimize()
    st.session_state.optimization_results = results
    st.session_state.last_tickers = selected_tickers
    st.session_state.computation_done = True

    UIComponents.render_optimization_status("completed")

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
        with st.spinner("📥 Loading market data..."):
            data_loader = get_data_loader()
            returns_df = data_loader.load_and_prepare_data(selected_tickers)
        UIComponents.render_data_preview(returns_df)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return

    # Optimize when button is clicked
    if optimize_btn:
        perform_optimization(selected_tickers, risk_aversion, optimization_case, returns_df)

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
        
        # Advanced Analytics
        render_advanced_visualizations(
            weights=results['weights'],
            tickers=st.session_state.last_tickers,
            returns_df=returns_df
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

        if len(active_weights) > 0:
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
        else:
            st.info("No assets with weight > 0.1%")

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

        # Convert TensorFlow tensors to floats
        clean_history = [{k: float(np.array(v)) if not isinstance(v, (int, float)) else float(v) for k, v in h.items()} for h in history]
        history_df = pd.DataFrame(clean_history)

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
