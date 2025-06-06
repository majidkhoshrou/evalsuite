# 📊 EvalSuite

**EvalSuite** is a lightweight Python toolkit designed for evaluating and analyzing forecast accuracy, managing time-indexed forecast data, and applying preprocessing utilities to improve robustness.

## 🚀 Features

- **Evaluation Metrics**  
  Implementations of standard forecast error metrics such as `MAE`, `MAPE`, `sMAPE`, `wMAPE`, and various relative error metrics.
- **Temporal Aggregation**  
  Evaluate metrics over different time frequencies (e.g., daily, quarterly) using `calculate_metrics(...)`.
- **Calendar-Based Evaluation**  
  Assess forecast accuracy across different calendar types (e.g., holidays, weekends) with `evaluate_calendar_types(...)`.
- **Data Cleaning Utilities**  
  Functions to clean and preprocess time series data, ensuring robustness in analysis.
- **Utility Functions**  
  Helpers for loading and aggregating forecasts, evaluating consistency between forecast iterations, merging and clipping DataFrames, and generating summaries.

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/majidkhoshrou/evalsuite.git
cd evalsuite
pip install -r requirements.txt
