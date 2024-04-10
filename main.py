# Python

# %%

import pandas as pd
from prophet import Prophet

# %%

df = pd.read_csv('_data/train.csv')
df['date'] = pd.to_datetime(df['date'])

#print(len(df[(df['family'] == 'PRODUCE') & (df['store_nbr'] == 3)].date.unique()));

store_sales = df[(df['family'] == 'PRODUCE') & (df['store_nbr'] == 3)].copy()

store_sales.drop('store_nbr', axis=1, inplace=True)
store_sales.drop('family', axis=1, inplace=True)
store_sales.drop('onpromotion', axis=1, inplace=True)
store_sales.drop('id', axis=1, inplace=True)

store_sales.columns = ['ds','y']
 
# %%
print(store_sales.tail())

# %%

m=Prophet(interval_width=0.95)
training_run = m.fit(store_sales)
# %%
future = m.make_future_dataframe(periods=100,freq='D')
forecast=m.predict(future)
forecast.tail()


# %%

plot1 = m.plot(forecast)


