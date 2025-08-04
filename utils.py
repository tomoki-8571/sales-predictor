import pandas as pd
import jpholiday
from lightgbm import LGBMRegressor
import plotly.graph_objs as go
import io
import base64

PERIOD_MAP = {
    "1d": 1,
    "1w": 7,
    "1m": 30,
    "3m": 90,
    "6m": 180,
    "1y": 365
}

def generate_features(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Weekday'] = df['Date'].dt.weekday
    df['Month'] = df['Date'].dt.month
    df['Holiday'] = df['Date'].apply(lambda x: int(jpholiday.is_holiday(x)))
    df['Lag1'] = df['Sales'].shift(1)
    df['MA3'] = df['Sales'].rolling(window=3).mean()
    df.dropna(inplace=True)
    return df

def train_and_predict(df, period_key):
    X = df[['Weekday', 'Month', 'Holiday', 'Lag1', 'MA3']]
    y = df['Sales']

    model = LGBMRegressor()
    model.fit(X, y)

    last_date = df['Date'].max()
    future_date = last_date + pd.Timedelta(days=PERIOD_MAP[period_key])
    future_df = pd.DataFrame({'Date': [future_date]})
    future_df['Weekday'] = future_date.weekday()
    future_df['Month'] = future_date.month
    future_df['Holiday'] = int(jpholiday.is_holiday(future_date))
    future_df['Lag1'] = df['Sales'].iloc[-1]
    future_df['MA3'] = df['Sales'].iloc[-3:].mean()

    X_future = future_df[['Weekday', 'Month', 'Holiday', 'Lag1', 'MA3']]
    prediction = model.predict(X_future)[0]

    # ▼ 予測結果DataFrame
    result_df = pd.DataFrame({
        'Date': [future_date],
        'PredictedSales': [prediction]
    })

    # ▼ グラフをPlotlyで作成
    fig = go.Figure()
    fig.add_trace(go.Bar(x=result_df['Date'], y=result_df['PredictedSales'], name='予測売上'))

    fig.update_layout(title='予測結果',
                      xaxis_title='日付',
                      yaxis_title='売上',
                      height=400)

    # ▼ HTML埋め込み用の画像生成（base64形式）
    img_bytes = fig.to_image(format="png")
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    # ▼ CSVダウンロード用の文字列（UTF-8）
    csv_buffer = io.StringIO()
    result_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    return result_df, img_base64, csv_data
