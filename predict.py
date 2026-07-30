import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler



MODEL_PATH = "stock_lstm_model.keras"

TIME_STEP = 100


model = load_model(MODEL_PATH)



def predict_stock(stock, start, end):



    df = yf.download(
        stock,
        start=start,
        end=end,
        auto_adjust=True
    )

    if df.empty or len(df) < TIME_STEP:
        return None, "Not enough data"


    df.columns = df.columns.get_level_values(0)


    # RSI
    df["RSI"] = ta.rsi(df["Close"], length=14)

    # MACD
    macd = ta.macd(df["Close"])

    df["MACD"] = macd.iloc[:, 0]

    df["MACD_SIGNAL"] = macd.iloc[:, 1]

    # EMA
    df["EMA20"] = ta.ema(df["Close"], length=20)

    # SMA
    df["SMA20"] = ta.sma(df["Close"], length=20)

    # Bollinger Bands
    bbands = ta.bbands(df["Close"], length=20)

    df["BB_UPPER"] = bbands.iloc[:, 0]

    df["BB_LOWER"] = bbands.iloc[:, 2]

    # ATR
    df["ATR"] = ta.atr(
        df["High"],
        df["Low"],
        df["Close"],
        length=14
    )


    df.dropna(inplace=True)



    features = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "RSI",
        "MACD",
        "MACD_SIGNAL",
        "EMA20",
        "SMA20",
        "BB_UPPER",
        "BB_LOWER",
        "ATR"
    ]

    data = df[features]



    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_data = scaler.fit_transform(data)


    X_input = scaled_data[-TIME_STEP:]



    X_input = X_input.reshape(
        1,
        TIME_STEP,
        len(features)
    )



    prediction = model.predict(X_input)



    dummy_array = np.zeros((1, len(features)))


    dummy_array[0, 3] = prediction[0][0]

    predicted_price = scaler.inverse_transform(
        dummy_array
    )[0][3]

    return round(predicted_price, 2), df