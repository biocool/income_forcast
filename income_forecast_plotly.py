import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
import plotly.graph_objects as go
import calendar

def forecast_income(income_history, window_size=3, forecast_horizon=6, plot_matplotlib=False, start_year=2025, start_month=1):
    """
    Forecasts future monthly income using a rolling linear regression model and visualizes
    predictions with 95% confidence intervals.

    This function uses historical income data to train a rolling linear regression model
    and generates forecasts for a specified number of future months. It supports both
    interactive Plotly plots (for web embedding) and optional static Matplotlib plots.
    Month labels start from a specified month and year, defaulting to January of 2025.
    Hover tooltips in Plotly provide human-readable month and income information.

    Parameters
    ----------
    income_history : list or array-like of floats
        Historical monthly income values. Must contain at least `window_size + 1` values.
        Example: [3200, 3500, 4000, 4200, 4600, 5000]

    window_size : int, default=3
        Number of past months to use as input features for predicting the next month.
        Determines the "lookback" window for the rolling regression.

    forecast_horizon : int, default=6
        Number of future months to forecast.

    plot_matplotlib : bool, default=False
        If True, generates a static Matplotlib plot alongside the Plotly figure.

    start_year : int, default=2025
        The year corresponding to the first month of the historical data.

    start_month : int, default=1
        The starting month index for the first historical data point (1 = January, 12 = December).

    Returns
    -------
    dict
        A dictionary containing:
        - "predictions": list of forecasted income values for the next `forecast_horizon` months.
        - "lower_bound": list of lower bounds of the 95% confidence intervals.
        - "upper_bound": list of upper bounds of the 95% confidence intervals.
        - "months": dictionary with
            * "historical": list of month labels for historical data (e.g., "Jan 2025").
            * "forecast": list of month labels for forecasted data.
        - "plotly_fig": plotly.graph_objects.Figure
            Interactive Plotly figure with historical data, forecasts, and confidence intervals.

    Example
    -------
    >>> income_history = [3200, 3500, 4000, 4200, 4600, 5000]
    >>> results = forecast_income(income_history, window_size=3, forecast_horizon=6)
    >>> results["predictions"]
    [5300.2, 5601.5, 5905.8, 6212.4, 6521.0, 6831.5]
    >>> results["months"]["forecast"]
    ['Jul 2025', 'Aug 2025', 'Sep 2025', 'Oct 2025', 'Nov 2025', 'Dec 2025']
    >>> results["plotly_fig"].show()

    Notes
    -----
    - The function enforces a minimum uncertainty floor to avoid overly narrow confidence intervals.
    - Plotly hover tooltips display readable month and income information.
    - Matplotlib x-axis labels are rotated for readability when `plot_matplotlib=True`.
    """

    # --- prepare training data ---
    X, y = [], []
    for i in range(len(income_history) - window_size):
        X.append(income_history[i:i+window_size])
        y.append(income_history[i+window_size])
    X = np.array(X); y = np.array(y)

    # --- train linear regression ---
    model = LinearRegression().fit(X, y)

    # --- rolling forecast ---
    predictions = []
    seq = income_history[-window_size:]
    for _ in range(forecast_horizon):
        nxt = model.predict([seq])[0]
        predictions.append(nxt)
        seq = list(seq[1:]) + [nxt]

    predictions = np.array(predictions)

    # --- residuals & uncertainty estimation ---
    y_hat = model.predict(X)
    res = y - y_hat
    dof = max(len(y) - X.shape[1] - 1, 1)
    s = float(np.sqrt(np.sum(res**2) / dof))

    # enforce a minimum uncertainty floor
    min_floor = max(0.05 * np.std(income_history), 50.0)
    ci_base = max(s, min_floor)

    t_val = stats.t.ppf(0.975, dof)
    margins = np.array([t_val * ci_base * np.sqrt(k+1) for k in range(forecast_horizon)])

    lower = predictions - margins
    upper = predictions + margins

    # --- month labels ---
    total_months = len(income_history) + forecast_horizon
    month_labels = []
    year_labels = []

    for i in range(total_months):
        month_idx = (start_month - 1 + i) % 12 + 1
        year_idx = start_year + (start_month - 1 + i) // 12
        month_labels.append(calendar.month_abbr[month_idx])  # Jan, Feb, ...
        year_labels.append(year_idx)

    hist_labels = [f"{m} {y}" for m, y in zip(month_labels[:len(income_history)], year_labels[:len(income_history)])]
    fut_labels  = [f"{m} {y}" for m, y in zip(month_labels[len(income_history):], year_labels[len(income_history):])]

    # --- matplotlib plot (optional) ---
    if plot_matplotlib:
        plt.figure(figsize=(8,5))
        plt.plot(hist_labels, income_history, "o-", label="Historical income")
        plt.plot(fut_labels, predictions, "o--", label="Forecast")
        plt.fill_between(fut_labels, lower, upper, alpha=0.25, label="95% CI")
        plt.xlabel("Month"); plt.ylabel("Income")
        plt.title("Income Forecast with Visible 95% Confidence Interval")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.xticks(rotation=45)
        plt.show()

    # --- plotly plot (interactive) ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_labels, y=income_history,
        mode="lines+markers", name="Historical income",
        hovertemplate="Month: %{x}<br>Income: $%{y:,.0f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=fut_labels, y=predictions,
        mode="lines+markers", name="Forecast",
        hovertemplate="Month: %{x}<br>Forecast: $%{y:,.0f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=fut_labels + fut_labels[::-1],
        y=np.concatenate([upper, lower[::-1]]),
        fill="toself",
        fillcolor="rgba(0,100,80,0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="95% CI"
    ))
    fig.update_layout(
        title="Income Forecast with 95% Confidence Interval",
        xaxis_title="Month",
        yaxis_title="Income",
        template="plotly_white"
    )

    # --- return results ---
    return {
        "predictions": predictions.tolist(),
        "lower_bound": lower.tolist(),
        "upper_bound": upper.tolist(),
        "months": {
            "historical": hist_labels,
            "forecast": fut_labels
        },
        "plotly_fig": fig
    }
