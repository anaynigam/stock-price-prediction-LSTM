import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd

from predict import predict_stock


def run_prediction(stock, start, end):

    result, df = predict_stock(
        stock,
        start,
        end
    )


    if df is None:
        return result, None

    predicted_price = float(result)

    stock=stock.upper().strip()

    if stock.endswith(".NS") or stock.endswith(".BO"):
        currency="₹"
        currency_name="INR"
    else:
        currency="$"
        currency_name="USD"



    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)



    last_date = df.index[-1]

    next_date = last_date + pd.offsets.BDay(1)



    last_close = float(df["Close"].iloc[-1])



    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Historical prices
    ax1.plot(
        df.index,
        df["Close"],
        label="Historical Close Price"
    )

    # Prediction point
    ax1.scatter(
        next_date,
        predicted_price,
        label="Predicted Next Close",
        zorder=5
    )

    # Dashed connection line
    ax1.plot(
        [last_date, next_date],
        [last_close, predicted_price],
        linestyle="--",
        alpha=0.7
    )



    ax1.set_title(f"{stock} Stock Price Prediction")

    ax1.set_xlabel("Date")

    ax1.set_ylabel(f"Price({currency}")

    ax1.legend()

    ax1.grid(True)


    fig_indicators,ax2=plt.subplots(figsize=(12, 6))
    ax2.plot(
        df.index,
        df["Close"],
        label="Closing Price",
    )
    ax2.plot(
        df.index,
        df['EMA20'],
        label="EMA20",
    )
    ax2.plot(
        df.index,
        df['SMA20'],
        label="SMA20",
    )
    ax2.plot(
        df.index,
        df["BB_LOWER"],
        label="BB_LOWER",
    )

    ax2.set_title(f"{stock} - Technical Indicators")
    ax2.set_xlabel("Date")
    ax2.set_ylabel(f"Price ({currency})")
    ax2.legend()
    ax2.grid(True)




    difference = predicted_price - last_close

    percentage_change = (
        difference / last_close
    ) * 100

    change_sign="+" if difference>=0 else ""

    output_text = f"""
Predicted Next Close Price: {currency}{predicted_price:.2f}{currency_name}

Last Close Price: {currency}{last_close:.2f}{currency_name}

Predicted Change: {currency}{difference:.2f}

Percentage Change: {percentage_change:.2f}%
"""

    return output_text, fig, fig_indicators



with gr.Blocks() as demo:

    gr.Markdown(
        """

### Multivariate LSTM + Technical Indicators

Features Used:
- OHLCV Data
- RSI
- MACD
- EMA
- SMA
- Bollinger Bands
- ATR

⚠️ Educational and research purposes only.
This is NOT financial advice.
"""
    )



    stock = gr.Textbox(
        label="Stock Symbol",
        value="MSFT"
    )

    start = gr.Textbox(
        label="Start Date (YYYY-MM-DD)",
        value="2000-01-01"
    )

    end = gr.Textbox(
        label="End Date (YYYY-MM-DD)",
        value="2026-10-01"
    )



    predict_btn = gr.Button("Predict")



    output_text = gr.Textbox(
        label="Prediction Results"
    )

    output_plot = gr.Plot(
        label="Prediction Graph"
    )

    output_plot_indicators = gr.Plot(
        label="Technical Indicators"
    )


    predict_btn.click(
        fn=run_prediction,
        inputs=[stock, start, end],
        outputs=[output_text, output_plot, output_plot_indicators]
    )

import os

if __name__=="__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT",7860))
    )