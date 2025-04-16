import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.tsa.api import Holt
from sklearn.linear_model import LinearRegression


# Load your dataset
df = pd.read_csv("water_consumption.csv")


# Streamlit layout
st.title("Clean Water Consumption Predictor")
st.write("### 💧Water: the only drink that falls from the sky for free.")
st.write("##### Every drop counts! Curious how much water we’ll use? Let’s find out together!")


# Dropdown selections
years = sorted(df['date'].unique())
states = sorted(df['state'].unique())
categories = ['All'] + sorted(df['sector'].unique())


selected_state = st.selectbox("Select State", states)
selected_category = st.selectbox("Select Sector", categories)


# Filter data
if selected_category == 'All':
    temp_df = df[df['state'] == selected_state]
    grouped_df = temp_df.groupby(['date'], as_index=False)['value'].sum()
    grouped_df['sector'] = 'All'
    filtered_df = grouped_df[['date', 'sector', 'value']]
else:
    filtered_df = df[
        (df['state'] == selected_state) &
        (df['sector'] == selected_category)
    ][['date', 'sector', 'value']]


if filtered_df.empty:
    st.warning("No data available for this combination. Please choose another state or sector.")
    st.stop()


# Preprocess
filtered_df = filtered_df.sort_values('date')
filtered_df['date'] = pd.to_numeric(filtered_df['date'])


# Train-test split (e.g., last 5 years for testing)
split_year = filtered_df['date'].max() - 5
train = filtered_df[filtered_df['date'] <= split_year]
test = filtered_df[filtered_df['date'] > split_year]


# Fit Holt’s Linear Trend Model
fit = Holt(np.asarray(train['value'])).fit(smoothing_level=0.3, smoothing_slope=0.1)
forecast_vals = fit.forecast(len(test))


# Merge forecast with test for prediction display
y_hat_avg = test.copy()
y_hat_avg['forecast'] = forecast_vals


# Fit Linear Regression Model
X_train = train['date'].values.reshape(-1, 1)
y_train = train['value'].values


linear_model = LinearRegression()
linear_model.fit(X_train, y_train)


# Predict future using Linear Regression
X_all = filtered_df['date'].values.reshape(-1, 1)
filtered_df['linear_forecast'] = linear_model.predict(X_all)


# Prediction slider
selected_year = st.slider("Select a Year", min_value=min(years), max_value=2040, value=2025)
last_year = filtered_df['date'].max()


# Predict for selected year
X_future = np.array([[selected_year]])
predicted_lr = linear_model.predict(X_future)[0]


if selected_year <= last_year:
    try:
        predicted_value_holt = filtered_df[filtered_df['date'] == selected_year]['value'].values[0]
    except IndexError:
        predicted_value_holt = np.nan
else:
    future_steps = selected_year - last_year
    predicted_value_holt = float(fit.forecast(future_steps)[-1])


# Display both predictions
st.success(f"🟢 Holt's forecast for {selected_year}: **{predicted_value_holt:.2f} million litres**")
st.success(f"🔵 Linear Regression forecast for {selected_year}: **{predicted_lr:.2f} million litres**")


fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(train['date'], train['value'], label='Train', color='blue')
ax.plot(test['date'], test['value'], label='Test', color='orange')
ax.plot(y_hat_avg['date'], y_hat_avg['forecast'], label='Holt Forecast', color='green', linestyle='--')
ax.plot(filtered_df['date'], filtered_df['linear_forecast'], label='Linear Regression', color='purple', linestyle='--')


if selected_year > last_year:
    ax.scatter(selected_year, predicted_value_holt, color='green', s=100, marker='o', label=f'{selected_year} Holt Prediction')
    ax.scatter(selected_year, predicted_lr, color='purple', s=100, marker='x', label=f'{selected_year} Linear Prediction')


ax.set_xlabel('Year')
ax.set_ylabel('Water Consumption (million litres)')
ax.set_title('Forecasting Water Consumption')
ax.legend(loc='best')
ax.grid(True)


st.pyplot(fig)
