Stock Price Prediction Using Multivariate LSTM

A deep learning-based stock price forecasting system that uses historical market data and technical indicators to predict the next closing price of a stock.

The project uses a **multivariate Long Short-Term Memory (LSTM)** neural network to learn temporal patterns from OHLCV data and engineered technical indicators. A **Gradio interface** allows users to enter a stock symbol and date range and visualize the resulting prediction.

> **Disclaimer:** This project is intended for educational and research purposes only. Predictions should not be considered financial advice.

---

## 🎯 Project Objective

The initial version of this project predicted the next closing price using only historical closing prices.

The model was later extended into a multivariate forecasting system incorporating:

- Open, High, Low, Close and Volume (OHLCV)
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- MACD Signal
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Bollinger Bands
- Average True Range (ATR)

The objective is to study whether additional market features and technical indicators can improve time-series forecasting compared with using closing prices alone.

---

## 🧠 Model Architecture

The forecasting model uses an LSTM neural network designed for sequential data.

```text
Historical Stock Data
        ↓
Technical Indicator Calculation
        ↓
Feature Scaling
        ↓
100-Day Sequences
        ↓
LSTM Layer
        ↓
Dropout
        ↓
LSTM Layer
        ↓
Dropout
        ↓
Dense Layers
        ↓
Next Closing Price
```

Each input sample contains the previous **100 trading observations** across all selected features.

---

## 📊 Technical Indicators

The model currently uses several indicators representing different characteristics of market behaviour.

| Indicator | Purpose |
|---|---|
| RSI | Measures price momentum |
| MACD | Captures momentum and trend changes |
| MACD Signal | Signal line associated with MACD |
| SMA | Represents longer-term price trend |
| EMA | Moving average giving greater weight to recent prices |
| Bollinger Bands | Represents price position relative to recent volatility |
| ATR | Measures market volatility |

---

## 📈 Model Performance

The multivariate LSTM produced the following results on the test data:

| Metric | Result |
|---|---:|
| R² Score | **0.9647** |
| MAE | **4.10** |
| MSE | **28.21** |
| RMSE | **5.31** |
| MAPE | **3.14%** |

The R² value indicates that the model closely tracks the variation in closing prices within the evaluated test period. It should **not** be interpreted as 96.47% prediction accuracy.

Financial time-series performance can vary substantially across stocks, time periods and market regimes.

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- yfinance
- pandas-ta
- Matplotlib
- Gradio

---

## 📁 Project Structure

```text
stock-lstm-prediction/
│
├── train.py              # Data preprocessing and LSTM training
├── predict.py            # Prediction pipeline
├── interface.py          # Gradio user interface
├── requirements.txt      # Python dependencies
├── .gitignore
└── README.md
```

Trained model files are excluded from version control and can be generated using the training pipeline.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd stock-lstm-prediction
```

Create and activate a virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Train the model

```bash
python train.py
```

### Launch the Gradio application

```bash
python interface.py
```

The interface allows the user to enter:

- Stock ticker
- Start date
- End date

and displays the predicted next closing price along with historical price data.

---

## 🔬 Current Development

The project is being developed incrementally to study different approaches to financial time-series modelling.

Current work focuses on comparing:

```text
Close Only
    ↓
OHLCV
    ↓
OHLCV + Technical Indicators
```

Future experiments may include directional prediction, additional feature engineering and strategy backtesting.

---

## ⚠️ Limitations

Stock prices are influenced by many factors that are not represented by historical OHLCV data or technical indicators, including news, macroeconomic conditions, earnings announcements and market sentiment.

A high forecasting R² therefore does not imply that the model can consistently generate profitable trading decisions.

This project should be treated as a **machine learning and time-series forecasting experiment**, not as an automated trading system.

---

## 👨‍💻 Author

**Anay**

B.Tech Computer Science & Engineering  
Specialization in Data Science
