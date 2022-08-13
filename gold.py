import pandas as pd
import numpy as np
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
df = pd.read_csv("gold_price.csv", parse_dates=True, index_col='Date')
df.head()



'''----------------------'''


df['Return'] = df['USD (PM)'].pct_change() * 100
df['Lagged_Return'] = df.Return.shift()
df = df.dropna()
train = df['2001':'2018']
test = df['2019']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train = train["Lagged_Return"].to_frame()
y_train = train["Return"]
X_test = test["Lagged_Return"].to_frame()
y_test = test["Return"]

model = LinearRegression ()
model.fit (X_train , y_train)
predictions= model.predict(x_test)

import matplotlib . pyplot as plt
out_of_sample_results = y_test.to_frame()
out_of_sample_results ["Out-of-Sample Predictions"] = model.predict(X_test)
out_of_sample_results.plot(subplots=True, Title='Gold prices, USD')
plt.show()








