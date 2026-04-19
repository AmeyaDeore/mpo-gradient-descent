# Multi-Objective Portfolio Optimization (MPO) with Gradient Descent

🎯 **A Real-Time Interactive Portfolio Optimization Platform**

This project implements an advanced portfolio optimization engine using **TensorFlow-driven Gradient Descent**. It allows users to dynamically balance risk and return through a premium web interface, leveraging modern financial mathematics and machine learning optimization techniques.

---

## 🚀 Key Features

- **Interactive Ticker Selection**: Choose up to 20 stocks from the S&P 500 for real-time analysis.
- **Dynamic Risk Aversion**: Adjust the $\lambda$ parameter via a slider to control the trade-off between expected returns and portfolio variance.
- **Advanced Optimization Strategies**:
  - **Case 1: Maximize Sharpe Ratio**: Optimizes for the best risk-adjusted return.
  - **Case 2: Minimize CVaR**: Focuses on extreme tail-risk reduction (95% Conditional Value at Risk).
  - **Case 3: UCITS Compliance**: Minimizes CVaR while enforcing regulatory constraints (max 10% weight per asset).
- **Premium Analytics Dashboard**:
  - High-fidelity **Dark Mode** UI with glassmorphism and neon-glow accents.
  - Real-time **Loss Convergence** charts tracking the gradient descent progress.
  - Interactive **Portfolio Allocation** (Pie & Bar charts).
  - Advanced **Risk-Return Scatter** and **Correlation Heatmaps**.
- **Data Export**: Download your optimized weights in CSV or JSON formats.

---

## 🛠️ Technology Stack

- **Presentation**: [Streamlit](https://streamlit.io/) (Premium Dark Theme)
- **Engine**: [TensorFlow 2.10](https://www.tensorflow.org/) (Gradient Tape for custom loss optimization)
- **Math**: [TensorFlow Probability](https://www.tensorflow.org/probability) (Percentile/Quantile calculations)
- **Data**: Pandas, NumPy, YFinance
- **Visualization**: [Plotly](https://plotly.com/) (Interactive subplots)

---

## ⚙️ Installation & Setup

This project requires **Python 3.10.6** for compatibility with TensorFlow 2.10. We recommend using `uv` for fast, reproducible environment management.

### 1. Prerequisites
Ensure you have the Python 3.10.6 toolchain installed. If using `uv`, you can install it automatically:
```powershell
# Using uv (Recommended)
uv python install cpython-3.10.6-windows-x86_64-none
```

### 2. Environment Setup
Create a dedicated virtual environment and install dependencies:
```powershell
# Create environment
uv venv .venv_310 --python cpython-3.10.6-windows-x86_64-none

# Activate (Windows)
.venv_310\Scripts\activate

# Install requirements
uv pip install -r requirements.txt
uv pip install tf-keras  # Required for modern TF compatibility
```

---

## 🏃 How to Run the App

Once the environment is set up and activated, start the Streamlit server:

```powershell
# Primary command
streamlit run src/streamlit_app.py
```

Or, run directly without manual activation:
```powershell
.venv_310\Scripts\streamlit.exe run src/streamlit_app.py
```

The app will be available at [http://localhost:8501](http://localhost:8501).

---

## 🧠 Architecture Overview

The system follows a **Three-Layer Architecture**:

1.  **Presentation Layer (`src/streamlit_app.py` & `src/ui_components.py`)**: Handles the user interface, session state, and interactive Plotly visualizations.
2.  **Logic Layer (`src/optimization_engine.py` & `src/models.py`)**: Implements the TensorFlow `MPOModel`. It uses a `GradientTape` to compute gradients of custom financial loss functions (Sharpe, CVaR) with respect to portfolio weights, optimizing them via the Adam optimizer.
3.  **Data Layer (`src/data_loader.py`)**: Loads S&P 500 historical data from local CSV files, calculates log returns, and prepares tensors for the engine.

### 🧮 Custom Loss Functions
The engine minimizes a composite loss:
$$L = -f_{\text{objective}}(w) + \lambda \cdot P(w)$$
Where $f$ is the target metric (e.g., Sharpe), $\lambda$ is the risk aversion, and $P(w)$ contains penalties for constraints like sparsity or UCITS limits.

---

## 🧪 Testing & Verification
A test suite is available to verify data loading and optimization logic:
```powershell
python test_streamlit_app.py
```

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

