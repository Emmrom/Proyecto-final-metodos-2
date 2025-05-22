

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pycoingecko import CoinGeckoAPI
import datetime
from scipy.interpolate import make_interp_spline
from statsmodels.tsa.holtwinters import ExponentialSmoothing

plt.style.use('seaborn-v0_8-darkgrid')
plt.figure(figsize=(16, 8))
# Obtener datos históricos de BTC
def get_btc_data():
    client = CoinGeckoAPI()
    start_time = int(datetime.datetime(2025, 3, 15).timestamp())
    end_time = int(datetime.datetime(2025, 3, 30).timestamp())

    try:
        data = client.get_coin_market_chart_range_by_id(
            id="bitcoin",
            vs_currency="usd",
            from_timestamp=start_time,
            to_timestamp=end_time
        )
        return data['prices']
    except Exception as e:
        print(f"Error al obtener datos: {e}")
        return None
# Procesamiento de datos
def process_data(prices):
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('date').resample('1h').mean().dropna()
    return df

# Modelado y extrapolación
def apply_spline(df, num_points=500):
    x = np.arange(len(df))
    y = df['price'].values
    spline = make_interp_spline(x, y, k=3)
    x_smooth = np.linspace(x.min(), x.max(), num_points)
    y_smooth = spline(x_smooth)
    date_smooth = pd.date_range(start=df.index[0], end=df.index[-1], periods=num_points)
    return pd.Series(y_smooth, index=date_smooth)

def spline_error(y_true, y_spline):
    return np.mean((y_true - y_spline)**2)  # MSE



def forecast_holt_winters(df, forecast_hours=24):
    model = ExponentialSmoothing(
        df['price'],
        trend='add',
        damped_trend=True,
        seasonal='add',
        seasonal_periods=24,
        initialization_method="estimated"
    ).fit(optimized=True)
    
    forecast = model.forecast(forecast_hours)
    return forecast, model

def plot_results(df, spline_series, forecast_series):
    plt.plot(df.index, df['price'], 'o', alpha=0.3, label='Datos originales')
    plt.plot(spline_series.index, spline_series.values, 'b-', label='Spline suavizado', linewidth=2)

    future_dates = forecast_series.index
    plt.plot(future_dates, forecast_series.values, 'r--', label='Predicción Holt-Winters', linewidth=2)
    plt.axvspan(spline_series.index[-1], future_dates[-1], color='orange', alpha=0.1)

    plt.title('Precio BTC: suavizado (Spline) + extrapolación (Holt-Winters)', fontsize=16, pad=20)
    plt.xlabel('Fecha', fontsize=12)
    plt.ylabel('Precio (USD)', fontsize=12)
    plt.legend(fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    btc_data = get_btc_data()
    if btc_data is None:
        exit()

    df = process_data(btc_data)
    if len(df) < 24:
        print("Datos insuficientes para modelado")
        exit()

    spline_series = apply_spline(df)
    forecast_series, model = forecast_holt_winters(df)

    plot_results(df, spline_series, forecast_series)

    print("\nMétricas del modelo Holt-Winters:")
    print(f"SSE: {model.sse:.2f}")
    print(f"AIC: {model.aic:.2f}")
    print(f"Parámetros: {model.params}")
