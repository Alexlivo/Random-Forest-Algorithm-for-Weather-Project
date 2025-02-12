import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Import datasets
pd.set_option('future.no_silent_downcasting', True)
weather = pd.read_csv('Sutton_Bonington_Dataset.csv', na_values='---')

# Delete null values
value_to_del = weather.loc["1959":"1960"]
weather = weather.drop(value_to_del.index)
weather = weather.drop(columns=["sun", "af", "rain"])

# Create new Dataframe
core_weather = weather
core_weather.columns = ['year', 'month', 'temp_max', 'temp_min']


# Convert core_weather index to DatetimeIndex
core_weather["dateInt"] = core_weather["year"].astype(str) + core_weather["month"].astype(str)
core_weather["date"] = pd.to_datetime(core_weather["dateInt"], format="%Y%m")
core_weather.index = core_weather["date"]
core_weather.index = core_weather.index.strftime("%Y-%m")
print(core_weather.drop(columns=["year", "month", "dateInt", "date"]))

# Set features
x = core_weather['temp_min']
y = core_weather['temp_max']

# Split data in training/testing years
train_data = core_weather.loc[:"1994"]
test_data = core_weather.loc["1995":]

x_train = train_data['temp_min']
y_train = train_data['temp_max'].values.reshape(-1, 1)
x_test = test_data['temp_min']
y_test = test_data['temp_max'].values.reshape(-1, 1)

# Initialise and train Random Forest Algorithm
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(y_train, x_train)

# Prediction on the test set
x_pred = rf.predict(y_test)

# Evaluation of model
mae = mean_absolute_error(x_test, x_pred)
r2 = r2_score(x_test, x_pred)

# Create new dataframe to compare actual and predicted values in table form
combined = pd.concat([x_test, pd.Series(x_pred.round(1), index=x_test.index)], axis=1)
combined.columns = ["Actual", "Predicted"]

print(f"Mean Absolute Error: {mae}")
print(f"R-Squared Score: {r2}")
print(f"Table Showing Actual vs Predicted Values:\n {combined}")

# Scatter Graph Plot
plt.figure(figsize=(16, 10))

plt.scatter(x_test.index, x_test, color='blue', label='Actual')
plt.scatter(x_test.index, x_pred, color='red', label='Predicted')

plt.title('Min Temperature Prediction using Random Forest')
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.xticks(x_test.index.unique()[::12], rotation=45, ha='right')

plt.legend()
plt.show()

# PSD Plot
plt.figure(figsize=(12, 6))

plt.psd(combined["Actual"], color="blue", label="Actual")
plt.psd(combined["Predicted"], color="red", label="Predicted")

plt.title("Power Spectral Density of Actual vs Minimum Predicted Temperature")
plt.xlabel("Frequency (cycles/year)")
plt.ylabel("PSD")
plt.xticks(rotation=0)

plt.legend()
plt.show()

# Convert DataFrame to csv
combined.to_csv("Min_Random_Forest_Dataset.csv")