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



    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)



    last_date = df.index[-1]

    next_date = last_date + pd.Timedelta(days=1)



    last_close = float(df["Close"].iloc[-1])



    fig, ax = plt.subplots(figsize=(12, 6))

    # Historical prices
    ax.plot(
        df.index,
        df["Close"],
        label="Historical Close Price"
    )

    # Prediction point
    ax.scatter(
        next_date,
        predicted_price,
        label="Predicted Next Close",
        zorder=5
    )

    # Dashed connection line
    ax.plot(
        [last_date, next_date],
        [last_close, predicted_price],
        linestyle="--",
        alpha=0.7
    )



    ax.set_title(f"{stock} Stock Price Prediction")

    ax.set_xlabel("Date")

    ax.set_ylabel("Price")

    ax.legend()

    ax.grid(True)



    difference = predicted_price - last_close

    percentage_change = (
        difference / last_close
    ) * 100

    output_text = f"""
Predicted Next Close Price: ${predicted_price:.2f}

Last Close Price: ${last_close:.2f}

Predicted Change: {difference:.2f}

Percentage Change: {percentage_change:.2f}%
"""

    return output_text, fig



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
        value="GOOG"
    )

    start = gr.Textbox(
        label="Start Date (YYYY-MM-DD)",
        value="2018-01-01"
    )

    end = gr.Textbox(
        label="End Date (YYYY-MM-DD)",
        value="2025-01-01"
    )



    predict_btn = gr.Button("Predict")



    output_text = gr.Textbox(
        label="Prediction Results"
    )

    output_plot = gr.Plot(
        label="Prediction Graph"
    )


    predict_btn.click(
        fn=run_prediction,
        inputs=[stock, start, end],
        outputs=[output_text, output_plot]
    )

demo.launch(share=True)