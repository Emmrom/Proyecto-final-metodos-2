import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pycoingecko import CoinGeckoAPI
import datetime
import time

# Configuración
POLY_DEGREE = 3  # Puedes cambiar este valor
EXTRAPOLATION_HOURS = 24

# Obtener datos de CoinGecko
def get_btc_data():
    client = CoinGeckoAPI()
    start_time = int(datetime.datetime(2025, 3, 15).timestamp())
    end_time = int(datetime.datetime(2025, 3, 30).timestamp())

    data = client.get_coin_market_chart_range_by_id(
        id="bitcoin",
        vs_currency="usd",
        from_timestamp=start_time,
        to_timestamp=end_time
    )
    return data['prices']

# Procesar datos
def process_data(prices):
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('date').resample('1h').mean().dropna()
    df['seconds'] = (df.index - df.index.min()).total_seconds()
    return df

# Mínimos cuadrados e interpolación/extrapolación
def least_squares_polyfit(df, degree, extrapolation_hours):
    x = df['seconds'].values
    y = df['price'].values

    coeffs = np.polyfit(x, y, deg=degree)
    poly = np.poly1d(coeffs)

    x_min, x_max = x.min(), x.max()
    x_extra = np.linspace(
        x_min - 3600 * extrapolation_hours,
        x_max + 3600 * extrapolation_hours,
        300
    )
    y_extra = poly(x_extra)
    base_time = df.index.min()
    time_extra = [base_time + datetime.timedelta(seconds=float(s)) for s in x_extra]
    return poly, time_extra, y_extra, x_min, x_max

# Graficar resultados
def plot_least_squares(df, time_extra, y_extra, x_extra_min, x_extra_max):
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['price'], 'o', label='Datos reales', markersize=4)

    # Clasificamos el rango interpolado vs extrapolado
    mask = (df.index.min() <= pd.to_datetime(time_extra)) & (pd.to_datetime(time_extra) <= df.index.max())
    time_extra = np.array(time_extra)

    plt.plot(time_extra, y_extra, label='Ajuste polinomial (Interpolación/Extrapolación)', color='green')
    
    # Marcar regiones
    plt.axvspan(time_extra[0], df.index.min(), color='red', alpha=0.1, label='Extrapolación')
    plt.axvspan(df.index.max(), time_extra[-1], color='red', alpha=0.1)

    plt.title(f"Interpolación y Extrapolación BTC usando Mínimos Cuadrados (grado {POLY_DEGREE})")
    plt.xlabel("Fecha")
    plt.ylabel("Precio (USD)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Flujo principal
if __name__ == "__main__":
    prices = get_btc_data()
    df = process_data(prices)
    poly, time_extra, y_extra, x_min, x_max = least_squares_polyfit(df, POLY_DEGREE, EXTRAPOLATION_HOURS)
    plot_least_squares(df, time_extra, y_extra, x_min, x_max)
