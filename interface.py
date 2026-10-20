import gradio as gr
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import os
from predict import predict_stock

CSS = """
    .gradio-container {max-width: 1550px !important; margin:auto !important; padding:24px !important;}
    body, .gradio-container {background:#090f1d !important; color:#e5e7eb !important;}
    #header {background:linear-gradient(110deg,#14213b,#0e1729);border:1px solid #26354d;border-radius:16px;padding:18px 24px;}
    #sidebar {background:#111b2c;border:1px solid #26354d;border-radius:15px;padding:18px;}
    .panel {background:#111b2c;border:1px solid #26354d;border-radius:15px;padding:16px;}
    .kpi {background:#142138;border:1px solid #293b54;border-radius:12px;padding:17px;min-height:117px;}
    .kpi-label {font-size:12px;color:#9aacc5;font-weight:600;letter-spacing:.3px;}
    .kpi-value {font-size:clamp(20px,2.2vw,30px);font-weight:750;color:#f8fafc;margin:8px 0 5px;overflow-wrap:anywhere;}
    .kpi-note {font-size:11px;color:#8494aa;}
    .kpi-value.up {color:#34d399;} .kpi-value.down {color:#fb7185;}
    .section-title {font-size:17px;font-weight:700;color:#f8fafc;margin-bottom:3px;}
    .section-sub {font-size:12px;color:#94a3b8;}
    #predict-button {background:#10b981 !important;color:#06150f !important;border:0 !important;font-weight:750 !important;}
    #sidebar input {background:#0c1424 !important;color:#f8fafc !important;}
    #warning {background:#302514;border:1px solid #655026;border-radius:10px;padding:12px;color:#fcd34d;}
    #status {font-size:13px;color:#a9bad0;}
    footer {display:none !important;}
    @media(max-width:850px){.gradio-container{padding:10px !important;}}
    """

EMPTY_KPIS = [
    '<div class="kpi"><div class="kpi-label">LATEST CLOSE</div><div class="kpi-value">—</div><div class="kpi-note">Last available trading session</div></div>',
    '<div class="kpi"><div class="kpi-label">PREDICTED NEXT CLOSE</div><div class="kpi-value">—</div><div class="kpi-note">LSTM model estimate</div></div>',
    '<div class="kpi"><div class="kpi-label">PREDICTED CHANGE</div><div class="kpi-value">—</div><div class="kpi-note">Compared with latest close</div></div>',
    '<div class="kpi"><div class="kpi-label">PREDICTED CHANGE %</div><div class="kpi-value">—</div><div class="kpi-note">Estimated relative movement</div></div>',
]


def card(label, value, note, movement=None):
    # All interpolated values are generated from numeric calculations, not raw user text.
    klass = 'up' if movement is not None and movement >= 0 else 'down' if movement is not None else ''
    return f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value {klass}">{value}</div><div class="kpi-note">{note}</div></div>'


def style_axis(ax, title, currency):
    ax.set_facecolor('#111b2c')
    ax.figure.set_facecolor('#111b2c')
    ax.set_title(title, loc='left', color='#f1f5f9', fontsize=14, fontweight='bold', pad=18)
    ax.set_ylabel(f'Price ({currency})', color='#94a3b8')
    ax.tick_params(colors='#a6b6ca', labelsize=9)
    ax.grid(color='#334155', alpha=.42, linestyle='--', linewidth=.7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.margins(x=.02)


def run_prediction(stock, start, end):
    # Original model, inputs, feature engineering and calculations are unchanged.
    try:
        result, df = predict_stock(stock, start, end)
    except Exception as exc:
        print(f'Prediction error: {exc}')
        return (*EMPTY_KPIS, 'Prediction unavailable. Please check the ticker, dates, and data service.', None, None)

    if result is None or df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return (*EMPTY_KPIS,
                'Prediction unavailable. Yahoo Finance may be rate-limiting requests or the stock data could not be downloaded. Please try again later.',
                None, None)
    try:
        predicted_price = float(result)
    except (TypeError, ValueError):
        return (*EMPTY_KPIS, 'Prediction failed: model did not return a valid price.', None, None)

    stock = stock.upper().strip()
    if stock.endswith('.NS') or stock.endswith('.BO') or stock.startswith("^"):
        currency, currency_name = '₹', 'INR'
    else:
        currency, currency_name = '$', 'USD'

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    last_date = df.index[-1]
    next_date = last_date + pd.offsets.BDay(1)
    last_close = float(df['Close'].iloc[-1])
    difference = predicted_price - last_close
    percentage_change = (difference / last_close) * 100
    sign = '+' if difference >= 0 else ''

    fig, ax1 = plt.subplots(figsize=(12, 5.3), dpi=110)
    ax1.plot(df.index, df['Close'], label='Historical close', color='#38bdf8', linewidth=2)
    ax1.scatter(next_date, predicted_price, label='Predicted next close',
                color='#34d399' if difference >= 0 else '#fb7185', s=95, zorder=5)
    ax1.plot([last_date, next_date], [last_close, predicted_price], linestyle='--', color='#fbbf24', linewidth=2,
             alpha=.9)
    style_axis(ax1, f'{stock}  ·  Historical Price & Forecast', currency)
    ax1.legend(loc='upper left', facecolor='#17253a', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=9)
    fig.tight_layout(pad=2)

    fig_indicators, ax2 = plt.subplots(figsize=(12, 5.3), dpi=110)
    ax2.plot(df.index, df['Close'], label='Close', color='#e2e8f0', linewidth=1.7)
    ax2.plot(df.index, df['EMA20'], label='EMA20', color='#38bdf8', linewidth=1.5)
    ax2.plot(df.index, df['SMA20'], label='SMA20', color='#fbbf24', linewidth=1.5)
    ax2.plot(df.index, df['BB_LOWER'], label='BB lower', color='#a78bfa', linewidth=1.3)
    style_axis(ax2, f'{stock}  ·  Technical Indicators', currency)
    ax2.legend(loc='upper left', facecolor='#17253a', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=9)
    fig_indicators.tight_layout(pad=2)

    cards = (
        card('LATEST CLOSE', f'{currency}{last_close:,.2f}', f'As of {last_date:%d %b %Y} · {currency_name}'),
        card('PREDICTED NEXT CLOSE', f'{currency}{predicted_price:,.2f}', 'Next-close LSTM estimate'),
        card('PREDICTED CHANGE', f'{sign}{currency}{difference:,.2f}', 'Compared with latest close', difference),
        card('PREDICTED CHANGE %', f'{sign}{percentage_change:.2f}%', 'Estimated relative movement', difference),
    )
    return (*cards,
            f'{stock} · Historical data through {last_date:%d %b %Y} · Next trading day estimated (weekends only; exchange holidays not checked).',
            fig, fig_indicators)


with gr.Blocks(theme=gr.themes.Soft(primary_hue='emerald', neutral_hue='slate'), css=CSS,
               title='MarketPulse | AI Stock Analytics') as demo:
    with gr.Row(elem_id='header'):
        gr.Markdown('## 📈 MarketPulse\n**AI STOCK ANALYTICS** · Multivariate LSTM + Technical Indicators')
    with gr.Row():
        with gr.Column(scale=1, min_width=240, elem_id='sidebar'):
            gr.Markdown('### Analysis Controls\nSelect a symbol and historical date range.')
            stock = gr.Textbox(label='Stock Symbol', value='MSFT', placeholder='MSFT or RELIANCE.NS')
            start = gr.Textbox(label='Start Date (YYYY-MM-DD)', value='2000-01-01')
            end = gr.Textbox(label='End Date (YYYY-MM-DD)', value='2026-10-01')
            predict_btn = gr.Button('Run Prediction →', variant='primary', elem_id='predict-button')
            gr.Markdown(
                '---\n### Model Details\n**Architecture:** Multivariate LSTM  \n**Lookback:** 100 observations  \n**Features:** OHLCV, RSI, MACD, EMA, SMA, Bollinger Bands, ATR')
            gr.HTML('<div id="warning">⚠️ Educational and research use only. Not financial advice.</div>')
        with gr.Column(scale=4, min_width=300):
            gr.Markdown(
                '<div class="section-title">Prediction Overview</div><div class="section-sub">Key results from your next-close forecasting model</div>')
            with gr.Row(equal_height=True):
                with gr.Column(min_width=165):
                    kpi_close = gr.HTML(EMPTY_KPIS[0])
                with gr.Column(min_width=165):
                    kpi_predicted = gr.HTML(EMPTY_KPIS[1])
                with gr.Column(min_width=165):
                    kpi_change = gr.HTML(EMPTY_KPIS[2])
                with gr.Column(min_width=165):
                    kpi_percent = gr.HTML(EMPTY_KPIS[3])
            status = gr.Markdown('Enter a stock symbol and click **Run Prediction**.', elem_id='status')
            with gr.Column(elem_classes=['panel']):
                gr.Markdown(
                    '<div class="section-title">Price Forecast</div><div class="section-sub">Historical closing prices and estimated next close</div>')
                output_plot = gr.Plot(show_label=False)
            with gr.Column(elem_classes=['panel']):
                gr.Markdown(
                    '<div class="section-title">Technical Analysis</div><div class="section-sub">Close, EMA20, SMA20 and lower Bollinger Band</div>')
                output_plot_indicators = gr.Plot(show_label=False)
    predict_btn.click(fn=run_prediction, inputs=[stock, start, end],
                      outputs=[kpi_close, kpi_predicted, kpi_change, kpi_percent, status, output_plot,
                               output_plot_indicators])

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=int(os.environ.get('PORT', 7860)))
