# EvalSuite

**EvalSuite** is a modular and extensible framework for evaluating time series forecasting models. It includes tools for:

- 🧹 **Cleaning and preprocessing** time series data  
- 📊 **Evaluating model accuracy** using common metrics  
- 📈 **Visualizing forecasts and residuals**  
- 💰 **Pricing model evaluation**

---

## 🚀 Installation

### Preferred (using [`uv`](https://github.com/astral-sh/uv)):

```bash
git clone https://github.com/majidkhoshrou/evalsuite.git
cd evalsuite
uv install
```

### Alternative (using `pip`):

```bash
git clone https://github.com/majidkhoshrou/evalsuite.git
cd evalsuite
pip install -r requirements.txt
```

---

## 🧰 Project Structure

```text
evalsuite/
│
├── datacleaning/        # Functions to clean and standardize time series data
├── metrics/             # Evaluation metrics: MAE, MAPE, RMSE, etc.
├── plotting/            # Visualization utilities for predictions and residuals
├── pricing/             # Tools to evaluate pricing model outputs
└── utils/               # Shared helpers
```

---

## 🔧 Usage

### Data Cleaning

```python
from evalsuite import datacleaning

cleaned_df = datacleaning.clean(df)
```

### Metrics Evaluation

```python
from evalsuite import metrics

mae = metrics.mean_absolute_error(y_true, y_pred)
```

### Plotting Results

```python
from evalsuite import plotting

plotting.plot_timeseries(df)
```

### Pricing Model Evaluation

```python
from evalsuite import pricing

result = pricing.evaluate_pricing_model(inputs)
```

---

## 🧪 Tests

To run the tests:

```bash
pytest
```

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create your feature branch (`git checkout -b my-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin my-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 📫 Contact

For questions or support, please open an issue or reach out to [@majidkhoshrou](https://github.com/majidkhoshrou).
