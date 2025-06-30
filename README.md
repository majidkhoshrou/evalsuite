# EvalSuite

**EvalSuite** is a modular, extensible framework for evaluating time series forecasting and pricing models. It includes tools for:

- 🧹 **Cleaning and preprocessing** time series data
- 📊 **Evaluating model accuracy** with a wide range of metrics
- 📈 **Visualizing forecasts and residuals**
- 💰 **Assessing pricing model performance**
- 📡 **Downloading and managing telemetry datasets**

---

## ✨ What's New in v0.2.0

✅ Refactored module structure for simpler imports  
✅ Improved CLI and configuration options  
✅ Added additional metrics (SMAPE, MASE)  
✅ Enhanced plotting with interactivity  
✅ **New `telemetry_downloading` module**  
✅ Dependency updates and performance improvements  

---

## 🚀 Installation

**Recommended (using [`uv`](https://github.com/astral-sh/uv)):**

```bash
git clone https://github.com/majidkhoshrou/evalsuite.git
cd evalsuite
uv pip install .
```

**Alternative (using pip):**

```bash
pip install git+https://github.com/majidkhoshrou/evalsuite@vx.x.x
```

Or if cloning locally:

```bash
git clone https://github.com/majidkhoshrou/evalsuite.git
cd evalsuite
pip install .
```

---

## 🧰 Project Structure

```
evalsuite/
├── datacleaning/           # Data cleaning and preprocessing
├── metrics/                # Evaluation metrics (MAE, MAPE, RMSE, SMAPE, MASE)
├── plotting/               # Visualization utilities
├── pricing/                # Pricing model evaluation
├── telemetry_downloading/  # Tools to download telemetry datasets
├── cli/                    # Command-line interface scripts
└── utils/                  # Shared helpers
```

---

## 🔧 Usage Examples

### Data Cleaning

```python
from evalsuite.datacleaning import clean

cleaned_df = clean(df)
```

### Metrics Evaluation

```python
from evalsuite.metrics import smape

smape_score = smape(y_true, y_pred)
```

### Plotting

```python
from evalsuite.plotting import plot_timeseries

plot_timeseries(df, title="Forecast vs Actual")
```

### Pricing Evaluation

```python
from evalsuite.pricing import evaluate

result = evaluate(inputs)
```

### Telemetry Data Downloading

```python
from evalsuite.telemetry_downloading import download_dataset

# Download and save telemetry data to a CSV
download_dataset(
    url="https://example.com/data.csv",
    output_path="data/telemetry.csv"
)
```

---

## ⚙️ Command-Line Interface

EvalSuite v0.2.0 adds experimental CLI support:

```bash
evalsuite-cli --help
```

Example:

```bash
evalsuite-cli evaluate --y_true data/true.csv --y_pred data/pred.csv --metric smape
```

---

## 🧪 Running Tests

To run the tests:

```bash
pytest
```

With coverage:

```bash
pytest --cov=evalsuite
```

---

## 🤝 Contributing

We welcome contributions!

1. Fork the repository
2. Create a feature branch (`git checkout -b my-feature`)
3. Commit your changes (`git commit -am "Add feature"`)
4. Push your branch (`git push origin my-feature`)
5. Open a Pull Request

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 📫 Contact

For questions or support, please [open an issue](https://github.com/majidkhoshrou/evalsuite/issues) or contact [@majidkhoshrou](https://github.com/majidkhoshrou).
