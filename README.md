# Income Forecast Function

This Python function `forecast_income` forecasts future income based on historical monthly data using a rolling linear regression model. It provides predictions, 95% confidence intervals, month labels starting from January, and an interactive Plotly chart ready to embed in a web application.

---

## Features

- Predicts future monthly income based on past data
- Calculates 95% confidence intervals for forecasts
- Automatically generates month labels starting from January
- Interactive Plotly figure for web embedding
- Optional static Matplotlib plot for quick visualization

---

## Dependencies

Install the required Python libraries:

```bash
pip install numpy matplotlib scikit-learn scipy plotly

Function Signature
forecast_income(
    income_history, 
    window_size=3, 
    forecast_horizon=6, 
    plot_matplotlib=False, 
    start_year=2025, 
    start_month=1
)
