from pycoingecko import CoinGeckoAPI

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import time

client = CoinGeckoAPI()

print(client.ping())

#Cambiar la fecha a formato que recibe la app
def unix_time(year, month, day, hour, second):
    date_time = datetime.datetime(year, month, day, hour, second)
    return time.mktime(date_time.timetuple())

# Cambiar las fechas a formato humano
def human_time(unix_time):
    return datetime.datetime.fromtimestamp(unix_time)

#Creamos funcion para calcular los ultimos 365 dias, pues es la version gratis
start_time = unix_time(2025, 3, 15, 0, 0)
end_time = unix_time(2025, 3, 30, 0, 0)
print("Start Time: ", start_time)
print("End Time: ", end_time)

#obtenemos la data de CoinGecko
coin_result = client.get_coin_market_chart_range_by_id(
    id = "bitcoin",
    vs_currency = "usd",
    from_timestamp = start_time,
    to_timestamp = end_time
)

#print("Keys", coin_result.keys())
#[Tiempo, Precio]
#print(coin_result)

#Vemos cuatos puntos de precio nos da
print("Coin prices: ", len(coin_result["prices"]))
#fecha del primer precio y del ultimo
print("First: ", coin_result["prices"][0][0], human_time(coin_result["prices"][0][0]/1000))
print("End: ", coin_result["prices"][0][0], human_time(coin_result["prices"][-1][0]/1000))

'''
El formato de que va a recibir la funcion de interpolacion
{
    "time": [1, 2, 3, ...],
    "price": [1000, 999, 1001, ...]
}
'''
coin = {}
coin["time"] = [x[0] for x in coin_result ["prices"]]
coin["price"] = [x[1] for x in coin_result["prices"]]
#print(coin)

coin_df = pd.DataFrame(coin)
coin_df["time"] = pd.to_datetime(coin_df["time"], unit="ms")

sns.lineplot(data=coin_df, x="time", y="price")
plt.title("Precio de Bitcoin (USD)")
plt.xlabel("Fecha")
plt.ylabel("Precio en USD")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#Fuente principal (CoinGecko)
#CoinGecko. (2025). Bitcoin (BTC) Price Chart - Historical Data. https://www.coingecko.com