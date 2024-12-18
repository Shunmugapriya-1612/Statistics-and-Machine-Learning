import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.ar_model import AutoReg
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("energy_consumption_data.csv")


# Converting the 'Timestamp' column to datetime format
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%d-%m-%Y %H:%M")

# Setting 'Timestamp' as the index
df.set_index('Timestamp', inplace=True)
df.index = pd.date_range(start=df.index[0], periods=len(df), freq='h')

# Ploting Energy_Usage over time
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Energy_Usage'], label='Energy Usage', color='blue')
plt.title('Energy Usage Over Time')
plt.xlabel('Date')
plt.ylabel('Energy Usage (kWh)')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Decomposing the time series (Additive or Multiplicative)
decomposition = seasonal_decompose(df['Energy_Usage'], model='additive', period=12)

# Extract Components
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

#Plotting the Components
plt.figure(figsize=(12, 10))

# Trend Component
plt.subplot(4, 1, 2)
plt.plot(trend, label='Trend', color='green')
plt.title('Trend Component')
plt.legend(loc='upper left')

# Seasonal Component
plt.subplot(4, 1, 3)
plt.plot(seasonal, label='Seasonality', color='orange')
plt.title('Seasonal Component')
plt.legend(loc='upper left')

# Residual Component
plt.subplot(4, 1, 4)
plt.plot(residual, label='Residual', color='red')
plt.title('Residual Component')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

scaler = MinMaxScaler()
time_series = df['Energy_Usage']
time_series_scaled = scaler.fit_transform(time_series.values.reshape(-1, 1)).flatten()

# Simple Moving Average (SMA) function
def calculate_sma(data, window):
    return data.rolling(window=window).mean()

# Weighted Moving Average (WMA) function
def calculate_wma(data, window):
    weights = np.arange(1, window + 1)
    return data.rolling(window=window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

# Parameters
window_size = 3  # Adjust the window size as needed

# Calculate SMA and WMA
time_series = df['Energy_Usage']
sma = calculate_sma(time_series, window_size)
wma = calculate_wma(time_series, window_size)

# Plotting the SMA and WMA
plt.figure(figsize=(10, 6))
plt.plot(time_series, label="Original Data", marker="o")
plt.plot(sma, label=f"SMA (Window={window_size})", linestyle="--")
plt.plot(wma, label=f"WMA (Window={window_size})", linestyle=":")
plt.title("Trend Estimation with SMA and WMA")
plt.xlabel('Date')
plt.ylabel('Energy Usage (kWh)')
plt.legend()
plt.grid()
plt.show()


# 1. Single Exponential Smoothing (SES)
def single_exponential_smoothing(data, smoothing_level=0.8):
    model = SimpleExpSmoothing(data)
    fitted_model = model.fit(smoothing_level=smoothing_level, optimized=False)
    return fitted_model.fittedvalues, fitted_model.forecast(steps=3)

# 2. Double Exponential Smoothing (DES)
def double_exponential_smoothing(data, trend="add", smoothing_level=0.8, smoothing_trend=0.2):
    model = ExponentialSmoothing(data, trend=trend, seasonal=None)
    fitted_model = model.fit(smoothing_level=smoothing_level, smoothing_trend=smoothing_trend, optimized=False)
    return fitted_model.fittedvalues, fitted_model.forecast(steps=3)

# 3. Triple Exponential Smoothing (Holt-Winters, TES)
def triple_exponential_smoothing(data, trend="add", seasonal="add", seasonal_periods=4):
    model = ExponentialSmoothing(data, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods,use_boxcox=True)
    fitted_model = model.fit(optimized=True)
    return fitted_model.fittedvalues, fitted_model.forecast(steps=3)

# Apply smoothing methods
smoothing_level = 0.8
smoothing_trend = 0.2
seasonal_periods = 12

ses_fitted, ses_forecast = single_exponential_smoothing(time_series, smoothing_level)
des_fitted, des_forecast = double_exponential_smoothing(time_series, "add", smoothing_level, smoothing_trend)
tes_fitted, tes_forecast = triple_exponential_smoothing(time_series, "add", "add", seasonal_periods=seasonal_periods)

# Plotting
plt.figure(figsize=(12, 8))
plt.plot(time_series, label="Original Data", marker="o")
plt.plot(ses_fitted, label="Single Exponential Smoothing (SES)", linestyle="--")
plt.plot(des_fitted, label="Double Exponential Smoothing (DES)", linestyle=":")
plt.plot(tes_fitted, label="Triple Exponential Smoothing (TES, Holt-Winters)", linestyle="-.")
plt.title("Exponential Smoothing Techniques")
plt.xlabel('Date')
plt.ylabel('Energy Usage (kWh)')
plt.legend()
plt.grid()
plt.show()

# Forecast Output
print("SES Forecast:", ses_forecast)
print("DES Forecast:", des_forecast)
print("TES Forecast:", tes_forecast)

#-------------------------------------AR------------------------------------------------------------
# Split the dataset into training and test sets
train_size = int(len(time_series_scaled) * 0.8)

time_series_scaled = pd.Series(time_series_scaled, index=time_series.index)
train, test = time_series_scaled[:train_size], time_series_scaled[train_size:]
#----------------------------------------------------AR------------------------------------------------------------
model = AutoReg(train, lags=2)  # AR(2) model
model_fit = model.fit()

# Make predictions
predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(time_series_scaled.index, time_series_scaled, label="Original Time Series")
plt.plot(test.index, predictions, label="Predictions", color="orange")
plt.axvline(x=train_size, color="red", linestyle="--", label="Train-Test Split")
plt.xlim(pd.to_datetime('2022-01-01'), df.index[-1])  # Set the x-axis to start from 2022
plt.xticks(rotation=45)  # Rotate the x-axis labels for better visibility
plt.legend()
plt.title("AR(2) Time Series Prediction")
plt.xlabel('Date')
plt.ylabel('Energy Usage (kWh)')
plt.grid()
plt.show()

# Print model summary
print(model_fit.summary())
#--------------------------------------------------------------------------------------------
# ==================== ARMA Model ==================== #
# Fit an ARMA(2, 2) model
# Uses the last two lagged values (p=2) and the last two error terms (q=2) for stationary data without differencing (d=0).
arma_model = ARIMA(train, order=(2, 0, 2))  # ARMA is ARIMA with d=0

arma_fit = arma_model.fit()
arma_predictions = arma_fit.predict(start=len(train), end=len(train) + len(test) - 1)

# ==================== ARIMA Model ==================== #
# Fit an ARIMA(2, 1, 2) model
arima_model = ARIMA(train, order=(2, 1, 2))  # ARIMA model with differencing
arima_fit = arima_model.fit()
arima_predictions = arima_fit.predict(start=len(train), end=len(train) + len(test) - 1, typ="levels")

# ==================== SARIMA Model ==================== #
# Fit a SARIMA(2, 1, 2)(1, 1, 1, 12) model
sarima_model = SARIMAX(train, order=(2, 1, 2), seasonal_order=(1, 1, 1, 12))
sarima_fit = sarima_model.fit()
sarima_predictions = sarima_fit.predict(start=len(train), end=len(train) + len(test) - 1)

# ==================== Plot Predictions ==================== #
plt.figure(figsize=(12, 6))
plt.plot(time_series_scaled.index, time_series_scaled, label="Original Time Series")
plt.plot(test.index, arma_predictions, label="ARMA Predictions", color="orange")
plt.plot(test.index, arima_predictions, label="ARIMA Predictions", color="green")
plt.plot(test.index, sarima_predictions, label="SARIMA Predictions", color="purple")
plt.axvline(x=train_size, color="red", linestyle="--", label="Train-Test Split")
plt.xlim(pd.to_datetime('2022-01-01'), df.index[-1])  # Set the x-axis to start from 2022
plt.xticks(rotation=45)  # Rotate the x-axis labels for better visibility
plt.legend()
plt.title("ARMA, ARIMA, SARIMA Predictions")
plt.xlabel('Date')
plt.ylabel('Energy Usage (kWh)')
plt.grid()
plt.show()

#---------------------------------------------------------------------------
# Function to calculate MAE and RMSE
def evaluate_model(true_values, predicted_values):
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    return mae, rmse

# Evaluate AR model
ar_mae, ar_rmse = evaluate_model(test, predictions)
print(f"AR Model - MAE: {ar_mae:.4f}, RMSE: {ar_rmse:.4f}")

# Evaluate ARMA model
arma_mae, arma_rmse = evaluate_model(test, arma_predictions)
print(f"ARMA Model - MAE: {arma_mae:.4f}, RMSE: {arma_rmse:.4f}")

# Evaluate ARIMA model
arima_mae, arima_rmse = evaluate_model(test, arima_predictions)
print(f"ARIMA Model - MAE: {arima_mae:.4f}, RMSE: {arima_rmse:.4f}")

# Evaluate SARIMA model
sarima_mae, sarima_rmse = evaluate_model(test, sarima_predictions)
print(f"SARIMA Model - MAE: {sarima_mae:.4f}, RMSE: {sarima_rmse:.4f}")

# Summarizing the best-performing model based on MAE and RMSE
models_performance = {
    "AR": {"MAE": ar_mae, "RMSE": ar_rmse},
    "ARMA": {"MAE": arma_mae, "RMSE": arma_rmse},
    "ARIMA": {"MAE": arima_mae, "RMSE": arima_rmse},
    "SARIMA": {"MAE": sarima_mae, "RMSE": sarima_rmse}
}

# Find the best model based on lowest MAE and RMSE
best_model_mae = min(models_performance, key=lambda x: models_performance[x]["MAE"])
best_model_rmse = min(models_performance, key=lambda x: models_performance[x]["RMSE"])

# Print the summary of the best model
print("\nModel Comparison Summary:")
print(f"Best Model based on MAE: {best_model_mae} (MAE: {models_performance[best_model_mae]['MAE']:.4f})")
print(f"Best Model based on RMSE: {best_model_rmse} (RMSE: {models_performance[best_model_rmse]['RMSE']:.4f})")

if best_model_mae == best_model_rmse:
    print(f"\nThe best model overall is: {best_model_mae} (Both MAE and RMSE)")
else:
    print(f"\nDifferent models performed best based on MAE and RMSE.")
    print(f"Best Model (MAE): {best_model_mae}")
    print(f"Best Model (RMSE): {best_model_rmse}")