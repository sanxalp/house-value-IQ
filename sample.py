import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
import joblib

# Load and inspect data
HouseDF = pd.read_csv("HousingData.csv")
HouseDF = HouseDF.reset_index()
print(HouseDF.head())
print(HouseDF.info())
print(HouseDF.describe())

# Visualization
sns.pairplot(HouseDF)
plt.show()

sns.displot(HouseDF['price'])
plt.show()

sns.heatmap(HouseDF.corr(), annot=True)
plt.show()

# Prepare data
X = HouseDF[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']]
y = HouseDF['price']  

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

# Scale the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler to a file
joblib.dump(scaler, 'scaler.pkl')  

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(X_train_scaled.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Reshape data to fit into the LSTM model
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# Train the model
model.fit(X_train_lstm, y_train, epochs=50)

# Make predictions
y_predicted = model.predict(X_test_lstm)

# Plot predictions
plt.scatter(y_test, y_predicted)
plt.show()

sns.displot((y_test - y_predicted.flatten()), bins=50)
plt.show()

# Performance metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_predicted))
print('MSE:', metrics.mean_squared_error(y_test, y_predicted))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_predicted)))

model.save('house_price_model.keras')
