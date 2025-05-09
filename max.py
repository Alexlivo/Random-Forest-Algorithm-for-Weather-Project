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
train_data = core_weather.loc[:"1989"]
test_data = core_weather.loc["1990":]

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
diff = (y_test - y_pred).round(1)
combined = pd.concat([y_test, pd.Series(y_pred.round(1), index=y_test.index), diff], axis=1)
combined.columns = ["Actual", "Predicted", "Difference"]

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

# Add month and year for seasonal grouping
combined["Month"] = combined.index.month
combined["Year"] = combined.index.year

# Assign season and adjust year for winter
def assign_season(row):
    month = row["Month"]
    year = int(row["Year"])
    if month in [3, 4, 5]:
        season = "Spring"
    elif month in [6, 7, 8]:
        season = "Summer"
    elif month in [9, 10, 11]:
        season = "Autumn"
    else:
        season = "Winter"
        # Shift year forward for winter (e.g., Dec 1995 becomes Winter 1996)
        if month == 12:
            year += 1
    return pd.Series([season, year])

combined[["Season", "Season_Year"]] = combined.apply(assign_season, axis=1)

# Create Season-Year label
combined["Season_Year"] = combined["Season_Year"].astype(int).astype(str) + " " + combined["Season"]

# Group by Season-Year and calculate mean
seasonal_year_avg = combined.groupby("Season_Year")[["Actual", "Predicted"]].mean().round(2)

# Sort by date using a helper datetime index
season_order = {"Winter": 1, "Spring": 2, "Summer": 3, "Autumn": 4}
seasonal_year_avg["SortKey"] = seasonal_year_avg.index.to_series().apply(
    lambda x: pd.to_datetime(x.split()[0] + f"-{season_order[x.split()[1]]*3:02d}-01")
)
seasonal_year_avg = seasonal_year_avg.sort_values("SortKey").drop(columns=["SortKey"])

# Choose the range or list of years you want to display
years_to_display = [2006, 2010, 2022]  # ← change as needed

# Extract year part from the index (which is like "2001 Spring")
seasonal_year_avg_filtered = seasonal_year_avg[
    seasonal_year_avg.index.to_series().str.extract(r'(\d{4})')[0].astype(int).isin(years_to_display)
]

# Plot grouped bar chart for selected years
plt.figure(figsize=(18, 8))

x = np.arange(len(seasonal_year_avg_filtered))
width = 0.4

plt.bar(x - width/2, seasonal_year_avg_filtered["Actual"], width, label='Actual', color='blue')
plt.bar(x + width/2, seasonal_year_avg_filtered["Predicted"], width, label='Predicted', color='red')

plt.title(f"Seasonal Max Temperature (Actual vs Predicted) for Selected Years: {years_to_display}")
plt.xlabel("Season-Year")
plt.ylabel("Temperature (°C)")
plt.xticks(x, seasonal_year_avg_filtered.index, rotation=45, size=14)
plt.grid(axis='y')
plt.legend()

plt.tight_layout()
plt.show()