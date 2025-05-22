from pycoingecko import CoinGeckoAPI
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import time
import numpy as np
from sklearn.linear_model import LinearRegression

# Configuración inicial 
client = CoinGeckoAPI()
print(client.ping())

def unix_time(year, month, day, hour, second):
    date_time = datetime.datetime(year, month, day, hour, second)
    return time.mktime(date_time.timetuple())

def human_time(unix_time):
    return datetime.datetime.fromtimestamp(unix_time)

start_time = unix_time(2025, 3, 15, 0, 0)
end_time = unix_time(2025, 3, 30, 0, 0)
print("Start Time: ", start_time)
print("End Time: ", end_time)

# Obtención de datos
coin_result = client.get_coin_market_chart_range_by_id(
    id="bitcoin",
    vs_currency="usd",
    from_timestamp=start_time,
    to_timestamp=end_time
)

print("Coin prices: ", len(coin_result["prices"]))
print("First: ", coin_result["prices"][0][0], human_time(coin_result["prices"][0][0]/1000))
print("End: ", coin_result["prices"][-1][0], human_time(coin_result["prices"][-1][0]/1000))

# Preparación de datos
coin = {
    "time": [x[0] for x in coin_result["prices"]],
    "price": [x[1] for x in coin_result["prices"]]
}
coin_df = pd.DataFrame(coin)
coin_df["time"] = pd.to_datetime(coin_df["time"], unit="ms")

# --- Gráfica 1: Original (línea continua) ---
plt.figure(figsize=(10, 5))
sns.lineplot(data=coin_df, x="time", y="price", color='blue')
plt.title("Precio de Bitcoin (USD) - Original")
plt.xlabel("Fecha")
plt.ylabel("Precio en USD")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- Gráfica 2: Línea de tendencia (regresión lineal) ---
# Convertir fechas a números para regresión
x = np.array(coin_df["time"].astype('int64') // 10**9).reshape(-1, 1)  # Unix timestamp
y = np.array(coin_df["price"])

# Ajustar modelo lineal
model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

plt.figure(figsize=(10, 5))
sns.lineplot(data=coin_df, x="time", y="price", color='blue', label='Datos reales')
plt.plot(coin_df["time"], y_pred, color='red', linestyle='--', label='Tendencia lineal')
plt.title("Precio de Bitcoin (USD) con Tendencia Lineal")
plt.xlabel("Fecha")
plt.ylabel("Precio en USD")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# --- Gráfica 3: Solo puntos (scatter plot) ---
plt.figure(figsize=(10, 5))
sns.scatterplot(data=coin_df, x="time", y="price", color='blue')
plt.title("Precio de Bitcoin (USD) - Puntos Discretos")
plt.xlabel("Fecha")
plt.ylabel("Precio en USD")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()