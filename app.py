# app.py
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

plt.style.use("seaborn-v0_8-darkgrid")


# ===============================
# Page config
# ===============================
st.set_page_config(page_title="GOOGLE Forecasting Dashboard", layout="wide")
st.title("GOOGL Dashboard: OLS, ARIMA & GARCH")


# ===============================
# Helpers
# ===============================
def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    # yfinance sometimes returns MultiIndex columns (Price, Ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df

def _pick_price_col(df: pd.DataFrame) -> str:
    # robust fallback if Adj Close not present
    if "Adj Close" in df.columns:
        return "Adj Close"
    if "Close" in df.columns:
        return "Close"
    raise KeyError(f"No 'Adj Close' or 'Close' in columns: {list(df.columns)}")


# ===============================
# Data loader (robust)
# ===============================
@st.cache_data(ttl=60 * 30)  # cache for 30 mins
def load_data(ticker: str):
    # Try standard first
    df = yf.download(ticker, period="5y", interval="1d", auto_adjust=False, progress=False)

    # Fallback: sometimes Adj Close is missing / data blocked
    if df is None or df.empty:
        df = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False)

    if df is None or df.empty:
        return None, None

    df = _flatten_yf_columns(df).dropna()

    # If auto_adjust=True, Adj Close often won't exist -> use Close
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    df["price"] = df[price_col].astype(float)
    df["returns"] = np.log(df["price"]).diff()

    df = df.dropna()
    return df, price_col



# ===============================
# Sidebar
# ===============================
st.sidebar.header("Model Settings")

ticker = st.sidebar.text_input("Ticker", value="GOOGL").strip().upper()
df, price_col = load_data(ticker)

if df is None:
    st.error(f"No data returned for {ticker}. Try another ticker or rerun.")
    st.stop()

# ===============================
# Market Snapshot (safe pre-fit metrics)
# ===============================
st.subheader("Market Snapshot")

last_price = float(df["price"].iloc[-1])
vol_20 = float(np.sqrt(252) * df["returns"].rolling(20).std().iloc[-1])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Last Price", f"${last_price:,.2f}")
c2.metric("20-Day Vol (ann.)", f"{vol_20*100:.2f}%")

# placeholders – filled after models fit
c3.metric("ARIMA RMSE", "—")
c4.metric("GARCH α+β", "—")


# OLS controls
st.sidebar.subheader("OLS")
ols_target = st.sidebar.selectbox("OLS target", ["price", "log(price)"], index=0)
ols_trend = st.sidebar.selectbox("Trend form", ["Linear", "Quadratic"], index=0)
ols_window_pct = st.sidebar.slider("Sample used for OLS (%)", 30, 100, 100)
show_ci = st.sidebar.checkbox("Show 95% confidence band", value=True)

# ARIMA controls
st.sidebar.subheader("ARIMA")
use_series = st.sidebar.selectbox("Series", ["price", "returns"], index=0)
p = st.sidebar.slider("p", 0, 3, 1)
d = st.sidebar.slider("d", 0, 2, 1)
q = st.sidebar.slider("q", 0, 3, 1)
order = (p, d, q)

# GARCH controls
st.sidebar.subheader("GARCH")
garch_p = st.sidebar.slider("GARCH p", 1, 3, 1)
garch_q = st.sidebar.slider("GARCH q", 1, 3, 1)
dist = st.sidebar.selectbox("Error distribution", ["normal", "t", "skewt"], index=1)
horizon = st.sidebar.slider("Volatility forecast horizon", 5, 60, 20)

# -----------------------
# Rolling ARIMA RMSE controls
# -----------------------
st.sidebar.subheader("Rolling ARIMA RMSE")
roll_window = st.sidebar.slider("Rolling window (obs)", 60, 400, 126, step=5)
refit_every = st.sidebar.slider("Refit every k steps", 1, 25, 5)
max_points = st.sidebar.slider("Max rolling points (speed)", 50, 400, 200, step=10)

# -----------------------
# GARCH regime colouring controls
# -----------------------
st.sidebar.subheader("GARCH Regimes")
regime_method = st.sidebar.selectbox("Regime split", ["Quantiles (33/66)", "Quantiles (25/75)"], index=0)


# ===============================
# Show data info
# ===============================
with st.expander("Data info"):
    st.write({"yfinance_price_column": price_col})
    st.write(df.tail())


# ===============================
# OLS trend model (interactive)
# ===============================
st.subheader("OLS Trend Model (Interactive)")

# subset window
n = len(df)
use_n = int(n * (ols_window_pct / 100))
df_ols = df.iloc[-use_n:].copy()

# target
if ols_target == "log(price)":
    y = np.log(df_ols["price"].astype(float))
    y_label = "log(price)"
else:
    y = df_ols["price"].astype(float)
    y_label = "price"

# design matrix
df_ols["t"] = np.arange(len(df_ols))

if ols_trend == "Quadratic":
    X = pd.DataFrame({"t": df_ols["t"], "t2": df_ols["t"] ** 2}, index=df_ols.index)
else:
    X = pd.DataFrame({"t": df_ols["t"]}, index=df_ols.index)

X = add_constant(X)

ols_fit = OLS(y, X).fit()
ols_pred = ols_fit.get_prediction(X)
ols_pred_mean = ols_pred.predicted_mean
ols_ci = ols_pred.conf_int(alpha=0.05)


# plot
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df_ols.index, y.values, label=f"Actual {y_label}")
ax.plot(df_ols.index, ols_pred_mean, linestyle="--", label="OLS fit")

if show_ci:
    ax.fill_between(df_ols.index, ols_ci[:, 0], ols_ci[:, 1], alpha=0.2, label="95% CI")


ax.set_title(f"OLS {ols_trend} Trend on last {ols_window_pct}% of data")
ax.legend()
st.pyplot(fig)

# metrics + summary
c1, c2, c3 = st.columns(3)
c1.metric("R²", f"{ols_fit.rsquared:.4f}")
c2.metric("Adj. R²", f"{ols_fit.rsquared_adj:.4f}")
c3.metric("AIC", f"{ols_fit.aic:.2f}")

with st.expander("OLS Model Summary"):
    st.text(ols_fit.summary())


# ===============================
# ARIMA model
# ===============================
st.subheader("ARIMA Backtest")

series = df[use_series].astype(float)

# returns are naturally business-day-ish; prices may have gaps; force business-day freq
series = series.asfreq("B").dropna()

cut = int(len(series) * 0.8)
train, test = series.iloc[:cut], series.iloc[cut:]

arima_fit = ARIMA(
    train,
    order=order,
    enforce_stationarity=False,
    enforce_invertibility=False
).fit()

forecast = arima_fit.get_forecast(steps=len(test))
arima_pred = pd.Series(np.asarray(forecast.predicted_mean), index=test.index)

mae = float(np.mean(np.abs(test.values - arima_pred.values)))
rmse = float(np.sqrt(np.mean((test.values - arima_pred.values) ** 2)))

arima_rmse = rmse


st.write({
    "Series": use_series,
    "Order": order,
    "MAE": round(mae, 6),
    "RMSE": round(rmse, 6),
})

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(train.index, train.values, label="Train")
ax.plot(test.index, test.values, label="Test")
ax.plot(arima_pred.index, arima_pred.values, label=f"ARIMA{order} Forecast")
ax.legend()
ax.set_title("ARIMA Forecast vs Actual")
st.pyplot(fig)


with st.expander("ARIMA Model Summary"):
    st.text(arima_fit.summary())

# ===============================
# Rolling ARIMA RMSE (1-step ahead)
# ===============================
st.subheader("Rolling ARIMA RMSE (1-step ahead)")

@st.cache_data(ttl=60 * 30)
def rolling_arima_rmse(series: pd.Series, order: tuple, window: int, refit_every: int, max_points: int) -> pd.Series:
    s = series.dropna().astype(float)
    if len(s) < window + 5:
        return pd.Series(dtype=float)

    # Sample a subset of evaluation points for speed (last max_points points)
    end = len(s) - 1
    start_eval = max(window, end - max_points)
    eval_idx = range(start_eval, end)

    preds = []
    actuals = []
    pred_index = []

    last_fit = None
    last_fit_i = None

    for i in eval_idx:
        train_slice = s.iloc[i - window:i]  # rolling window
        y_true = s.iloc[i]

        do_refit = (last_fit is None) or (last_fit_i is None) or ((i - last_fit_i) >= refit_every)

        if do_refit:
            try:
                last_fit = ARIMA(
                    train_slice,
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit()
                last_fit_i = i
            except Exception:
                last_fit = None

        if last_fit is None:
            continue

        try:
            f = last_fit.get_forecast(steps=1).predicted_mean.iloc[0]
        except Exception:
            continue

        preds.append(float(f))
        actuals.append(float(y_true))
        pred_index.append(s.index[i])

    if len(preds) < 5:
        return pd.Series(dtype=float)

    err = np.array(actuals) - np.array(preds)
    # rolling RMSE over the same 'window' length but on forecast errors
    rmse_series = pd.Series(err**2, index=pd.Index(pred_index)).rolling(min(window, len(err))).mean() ** 0.5
    rmse_series.name = "Rolling RMSE"
    return rmse_series.dropna()

roll_rmse = rolling_arima_rmse(series, order, roll_window, refit_every, max_points)

if roll_rmse.empty:
    st.info("Not enough data (or model failures) to compute rolling RMSE with current settings.")
else:
    st.metric("Latest rolling RMSE", f"{roll_rmse.iloc[-1]:.6f}")

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(roll_rmse.index, roll_rmse.values)
    ax.set_title(f"Rolling 1-step RMSE (window={roll_window}, refit_every={refit_every}, last {max_points} points)")
    ax.set_xlabel("Date")
    ax.set_ylabel("RMSE")
    st.pyplot(fig)


# ===============================
# GARCH model
# ===============================
st.subheader("GARCH Volatility Model")


# Use percent returns for stability in arch
returns_pct = (df["returns"].astype(float) * 100).dropna()

garch = arch_model(
    returns_pct,
    mean="Constant",
    vol="GARCH",
    p=garch_p,
    q=garch_q,
    dist=dist,
    rescale=False
)

garch_fit = garch.fit(disp="off")
# store for later display
garch_persistence = float(
    sum(v for k, v in garch_fit.params.items() if ("alpha" in k.lower()) or ("beta" in k.lower()))
)


col1, col2 = st.columns(2)
with col1:
    st.text(garch_fit.summary())

with col2:
    st.metric("Log-Likelihood", f"{garch_fit.loglikelihood:.2f}")
    st.metric("AIC", f"{garch_fit.aic:.2f}")
    st.metric("BIC", f"{garch_fit.bic:.2f}")
    st.metric("Obs", f"{garch_fit.nobs}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(garch_fit.conditional_volatility.index, garch_fit.conditional_volatility.values)
ax.set_title("Conditional Volatility (returns in % units)")
st.pyplot(fig)

# ===============================
# Regime colouring using GARCH volatility
# ===============================
st.subheader("Volatility Regimes (GARCH-coloured price)")

# Align conditional vol to df index
cond_vol = pd.Series(
    garch_fit.conditional_volatility,
    index=returns_pct.index
).reindex(df.index).dropna()

if len(cond_vol) < 50:
    st.info("Not enough volatility points to build regimes.")
else:
    if regime_method == "Quantiles (25/75)":
        q_low, q_high = cond_vol.quantile(0.25), cond_vol.quantile(0.75)
    else:
        q_low, q_high = cond_vol.quantile(0.33), cond_vol.quantile(0.66)

    # 0=Low, 1=Mid, 2=High
    regime = pd.Series(1, index=cond_vol.index)
    regime[cond_vol <= q_low] = 0
    regime[cond_vol >= q_high] = 2

    # Plot price with shaded regimes
    price_aligned = df["price"].reindex(regime.index)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(price_aligned.index, price_aligned.values, linewidth=1.5)

    # Shade contiguous regime blocks
    r = regime.values
    idx = regime.index
    start = 0
    for i in range(1, len(r) + 1):
        if i == len(r) or r[i] != r[i - 1]:
            # block [start, i)
            left = idx[start]
            right = idx[i - 1]
            code = r[i - 1]

            # default matplotlib cycle colors; just vary alpha
            if code == 0:
                ax.axvspan(left, right, alpha=0.10)
            elif code == 1:
                ax.axvspan(left, right, alpha=0.05)
            else:
                ax.axvspan(left, right, alpha=0.15)

            start = i

    ax.set_title("Price with GARCH volatility regimes (shaded)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    st.pyplot(fig)

    cA, cB, cC = st.columns(3)
    cA.metric("Low-vol threshold", f"{q_low:.3f}")
    cB.metric("High-vol threshold", f"{q_high:.3f}")
    cC.metric("Latest regime", ["Low", "Mid", "High"][int(regime.iloc[-1])])


st.subheader("Model Snapshot (after fitting)")

c1, c2 = st.columns(2)
c1.metric("ARIMA RMSE", f"{arima_rmse:.6f}")
c2.metric("GARCH α+β", f"{garch_persistence:.3f}")


# ===============================
# Volatility Forecast
# ===============================
st.subheader("GARCH Volatility Forecast")

fcst = garch_fit.forecast(horizon=horizon, reindex=False)
var_f = fcst.variance.values[-1, :]
vol_f = np.sqrt(var_f)

fig, ax = plt.subplots()
ax.plot(range(1, horizon + 1), vol_f)
ax.set_title("Forecasted Volatility (daily, % return units)")
ax.set_xlabel("Day")
ax.set_ylabel("Volatility")
st.pyplot(fig)
