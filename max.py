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

x_train = train_data['temp_min'].values.reshape(-1, 1)
y_train = train_data['temp_max']
x_test = test_data['temp_min'].values.reshape(-1, 1)
y_test = test_data['temp_max']

# Initialise and train Random Forest Algorithm
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)

# Prediction on the test set
y_pred = rf.predict(x_test)

# Evaluation of model
mae = mean_absolute_error(y_test, y_pred)
mae = round(mae, 3)
r2 = r2_score(y_test, y_pred)
r2 = round(r2, 3)

# Create new dataframe to compare actual and predicted values in table form
combined = pd.concat([y_test, pd.Series(y_pred.round(1), index=y_test.index)], axis=1)
combined.columns = ["Actual", "Predicted"]

# Ensure combined index is in DatetimeIndex format
combined.index = pd.to_datetime(combined.index, errors='coerce')

# Create new DataFrame combining actual and predicted values for averaging
yearly_avg = combined.resample("YE").mean().round(1)
yearly_avg.index = yearly_avg.index.strftime("%Y")

# Scatter Graph Plot
plt.rc('axes', titlesize=20)
plt.rc('axes', labelsize=18)

plt.figure(figsize=(16, 10))

plt.scatter(y_test.index, y_test, color='blue', label='Actual')
plt.scatter(y_test.index, y_pred, color='red', label='Predicted')

plt.title('Max Temperature Prediction using Random Forest')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.xticks(y_test.index.unique()[::12], rotation=45, ha='right')

plt.legend()
plt.show()

# Time Series Graph
plt.rc('axes', titlesize=20)
plt.rc('axes', labelsize=18)

plt.figure(figsize=(16, 8))

plt.plot(yearly_avg.index, yearly_avg["Actual"], color='blue', label='Actual')
plt.plot(yearly_avg.index, yearly_avg["Predicted"], color='red', label='Predicted')

plt.title("Time Series Graph of Max Temperature Prediction Comparison (RF)")
plt.xlabel("Year")
plt.ylabel("Temperature (°C)")
plt.xticks(yearly_avg.index, rotation=45, ha='right')

plt.grid(True)
plt.legend()
plt.show()

print(f"Mean Absolute Error: {mae}\n")
print(f"R-Squared Score: {r2}\n")
print(f"Table Showing Actual vs Predicted Values:\n {combined}\n")
print(f"Table showing average temps through year:\n {yearly_avg}")

# Convert DataFrame to csv
combined.to_csv("Max_Random_Forest_Dataset.csv")