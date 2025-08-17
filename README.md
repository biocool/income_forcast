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
```

## Function Signature
forecast_income(
    income_history, 
    window_size=3, 
    forecast_horizon=6, 
    plot_matplotlib=False, 
    start_year=2025, 
    start_month=1
)

## Parameters

income_history (list[float]): Historical monthly income values. Example: [3200, 3500, 4000]

window_size (int, default=3): Number of past months to use for predicting the next month

forecast_horizon (int, default=6): Number of months to forecast into the future

plot_matplotlib (bool, default=False): If True, shows a static Matplotlib plot

start_year (int, default=2025): Year of the first data point

start_month (int, default=1`): Starting month (1 = January, 12 = December)


## Returns

A dictionary with the following keys:

```python
{
    "predictions": [...],  # list of forecasted income values
    "lower_bound": [...],  # lower bounds of 95% confidence intervals
    "upper_bound": [...],  # upper bounds of 95% confidence intervals
    "months": {
        "historical": ["Jan 2025", "Feb 2025", ...],
        "forecast": ["Jul 2025", "Aug 2025", ...]
    },
    "plotly_fig": <plotly.graph_objects.Figure>  # interactive figure
}
```

## Example Usage
```python
import numpy as np
from forecast_module import forecast_income  # adjust import as needed

# Historical income data for Jan–Jun 2025
income_history = [3200, 3500, 4000, 4200, 4600, 5000]

# Generate 6-month forecast
results = forecast_income(
    income_history, 
    window_size=3, 
    forecast_horizon=6
)

# Print predicted values
print("Predictions:", results["predictions"])
print("Forecast months:", results["months"]["forecast"])

# Show interactive Plotly figure
results["plotly_fig"].show()

# Export Plotly figure as standalone HTML
results["plotly_fig"].write_html("forecast_plot.html", include_plotlyjs="cdn")
```
# Integration Notes

## Embed in a web page:

```python
html_code = results["plotly_fig"].to_html(full_html=False, include_plotlyjs="cdn")
# Insert html_code into your web frontend (React, Django, Flask, etc.)
```

## Standalone HTML report:

```python
results["plotly_fig"].write_html("forecast.html", include_plotlyjs="cdn")
```
