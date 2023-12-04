import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
 
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('C:/Users/TaylorJ/Documents/Python Stock Price Predictor/Python-Stock-History-Predictor2/stockdata/AAPL.csv')
df.tail()

df.shape

df.describe()

df.info()

plt.figure(figsize=(16,8))
plt.plot(df['Close'])
plt.title('Apple Close Price', fontsize=20)
plt.ylabel('Price in dollars')
plt.show()