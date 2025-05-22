import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pycoingecko import CoinGeckoAPI
import datetime
import time

client = CoinGeckoAPI()

# FUNCIONES DE TIEMPO
def unix_time(year, month, day, hour, second):
    date_time = datetime.datetime(year, month, day, hour, second)
    return time.mktime(date_time.timetuple())

# RANGO DE FECHAS
start_time = unix_time(2025, 3, 15, 0, 0)
end_time = unix_time(2025, 3, 30, 0, 0) 

# DATOS DE BTC
coin_result = client.get_coin_market_chart_range_by_id(
    id="bitcoin",
    vs_currency="usd",
    from_timestamp=start_time,
    to_timestamp=end_time
)

# CONVERTIR A DATAFRAME
coin = {
    "time": [x[0] for x in coin_result["prices"]],
    "price": [x[1] for x in coin_result["prices"]]
}

df = pd.DataFrame(coin)
df["time"] = pd.to_datetime(df["time"], unit="ms")

'''
AGRUPAR POR HORA EXACTA

Campos que acepta .resample()
"1H" -> cada hora
"30min" → cada 30 minutos
"1D" → cada día
"15min" → cada 15 minutos

Campos que acepta .mean() -> promedio
.first() → el primer precio de cada hora
.last() → el último precio de cada hora
.max() → el precio más alto por hora
.min() → el más bajo
'''
df_hourly = df.resample("1h", on="time").mean().reset_index()

df_hourly["seconds"] = (df_hourly["time"] - df_hourly["time"].min()).dt.total_seconds()

x = df_hourly["seconds"].values  # eje X numérico
y = df_hourly["price"].values    # precios

# Crear la función de interpolación (permite extrapolar también)
f_interp = interp1d(x, y, kind='cubic', fill_value='extrapolate')

# Valores para interpolar/extrapolar
extension_hours = 20
# Ajustar un polinomio de grado  a tus datos
coeffs = np.polyfit(x, y, deg=6)
poly = np.poly1d(coeffs)

# Crear nueva serie extendida
x_new = np.linspace(x[0] - 3600*extension_hours, x[-1] + 3600*extension_hours, 100)
y_new = poly(x_new)

# Convertimos x_new a fechas reales para graficar
start_time = df_hourly["time"].min()
x_new_dates = [start_time + datetime.timedelta(seconds=float(s)) for s in x_new]

# Graficamos
plt.figure(figsize=(12, 6))
plt.plot(df_hourly["time"], df_hourly["price"], 'o', label='Original')
plt.plot(x_new_dates, y_new, '-', label='Interpolación/Extrapolación')
plt.title("Interpolación y Extrapolación de precios BTC")
plt.xlabel("Tiempo")
plt.ylabel("Precio USD")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


'''
np.plyfit nos permite hacer una inteprolacion polinomica 
(ajuste de curva)
con deg=6 nos realiza la interpolacion con polinomio de grado 6
 
'''
