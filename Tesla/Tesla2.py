import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load data from CSV file
file_path = 'C:/Users/TaylorJ/Documents/Python Stock Price Predictor/Python-Stock-History-Predictor2/stockdata/TSLA (1).csv'
data = pd.read_csv(file_path)

# Ensure that the 'Date' column is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Use the index as the features (X) and closing prices as the target variable (y)
X = np.array(data.index).reshape(-1, 1)
y = data['Close'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='black', label='Actual Prices')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Predicted Prices')
plt.title('Tesla Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Closing Price')
plt.legend()
plt.show()
