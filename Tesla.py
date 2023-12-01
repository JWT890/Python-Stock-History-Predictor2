import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import math

data = pd.read_csv('C:/Users/TaylorJ/Documents/Python Stock Price Predictor/Python-Stock-History-Predictor2/stockdata/TSLA (1).csv')
data.head()
data.info()
data.describe()

X = data[['Open', 'High', 'Low', 'Volume', 'Close']].values
y = data[['Close']].values
print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
Model = LinearRegression()

Model.fit(X_train, y_train)
print(Model.coef_)
predicted = Model.predict(X_test)
print(predicted)

data1 = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': predicted.flatten()})
data1.head(20)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predicted))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predicted))
print('Root Mean Squared Error:', math.sqrt(metrics.mean_squared_error(y_test, predicted)))

graph = data1.head(20)
graph.plot(kind='bar')