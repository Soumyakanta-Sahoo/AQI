import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
import warnings

warnings.filterwarnings("ignore")


# ----------------- AQI Category Mapping -----------------
def categorize_aqi(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


# ----------------- Time Series Prep -----------------
def prepare_city_timeseries(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['City', 'Date'])
    city_aqi = df.groupby(['City', 'Date'])['AQI'].mean().reset_index()
    return city_aqi


# ----------------- Prophet Forecast -----------------
def forecast_with_prophet(city_aqi, city_name, periods=90):
    # Ensure output directory exists

    city_df = city_aqi[city_aqi['City'] == city_name][['Date', 'AQI']].dropna()
    city_df.columns = ['ds', 'y']

    model = Prophet()
    model.fit(city_df)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    # Add AQI category
    forecast['AQI_Category'] = forecast['yhat'].apply(categorize_aqi)

    # Export to CSV
    output_csv = f"output/forecast_{city_name.lower()}.csv"
    forecast.to_csv(output_csv, index=False)
    print(f"\n📄 Forecast saved to: {output_csv}")

    # Display preview
    print("\n🔮 Forecast preview:")
    print(forecast[['ds', 'yhat', 'AQI_Category']].tail(periods))

    # Main forecast plot
    fig = model.plot(forecast)
    plt.title(f"[Prophet] AQI Forecast for {city_name}")
    plt.xlabel("Date")
    plt.ylabel("AQI")
    plt.tight_layout()

    output_plot = f"output/forecast_{city_name.lower()}_plot.png"
    fig.savefig(output_plot)
    print(f"🖼️ Forecast plot saved to: {output_plot}")
    plt.show()

    # Components plot
    components_fig = model.plot_components(forecast)
    components_output = f"output/forecast_{city_name.lower()}_components.png"
    components_fig.savefig(components_output)
    print(f"📊 Components plot saved to: {components_output}")
    plt.show()


'''
# ----------------- ARIMA Forecast -----------------
def forecast_with_arima(city_aqi, city_name, steps=90):
    city_df = city_aqi[city_aqi['City'] == city_name].set_index('Date')['AQI'].dropna()

    try:
        model = ARIMA(city_df, order=(1, 1, 1))
        fitted = model.fit()
        forecast = fitted.forecast(steps=steps)

        if forecast.isna().sum() > 0:
            print("⚠️ ARIMA forecast contains NaN values. Try a different model order.")
            print(forecast.head())
            return

        forecast_index = pd.date_range(start=city_df.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')
        forecast_series = pd.Series(forecast.values, index=forecast_index)

        plt.figure(figsize=(12, 5))
        city_df.plot(label='Historical', color='blue')
        forecast_series.plot(label='Forecast (ARIMA)', color='red')

        plt.title(f"[ARIMA] AQI Forecast for {city_name}")
        plt.xlabel("Date")
        plt.ylabel("AQI")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"❌ ARIMA failed for {city_name}: {str(e)}")
'''


# ----------------- Seasonal Decomposition -----------------
def decompose_city_aqi(city_aqi, city_name):
    city_df = city_aqi[city_aqi['City'] == city_name].set_index('Date')['AQI'].dropna()
    result = seasonal_decompose(city_df, model='additive', period=365)

    result.plot()
    plt.suptitle(f"Seasonal Decomposition of AQI - {city_name}", fontsize=16)
    plt.tight_layout()
    plt.show()


'''
# ----------------- Prophet vs ARIMA -----------------
def compare_forecasts(city_aqi, city_name, steps=90):
    # Prophet
    prophet_df = city_aqi[city_aqi['City'] == city_name][['Date', 'AQI']].dropna()
    prophet_df.columns = ['ds', 'y']
    prophet_model = Prophet()
    prophet_model.fit(prophet_df)
    future = prophet_model.make_future_dataframe(periods=steps)
    prophet_forecast = prophet_model.predict(future)[['ds', 'yhat']].set_index('ds').iloc[-steps:]

    # ARIMA
    arima_series = city_aqi[city_aqi['City'] == city_name].set_index('Date')['AQI'].dropna()
    print(arima_series.tail(10))
    print("Missing values in ARIMA input:", arima_series.isna().sum())
    print("ARIMA last dates:", arima_series.index[-5:])

    try:
        print("🔄 Fitting auto_arima model...")
        auto_model = auto_arima(arima_series, seasonal=False, stepwise=True, suppress_warnings=True,
                                error_action='ignore')

        forecast_values = auto_model.predict(n_periods=steps)

        if pd.Series(forecast_values).isna().sum() > 0:
            print("⚠️ auto_arima forecast contains NaNs.")
            return

        forecast_index = pd.date_range(start=arima_series.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')
        arima_forecast = pd.Series(forecast_values, index=forecast_index)

        print("✅ ARIMA Forecast Preview:")
        print(arima_forecast.head())

        # Plotting
        plt.figure(figsize=(14, 6))
        arima_series.plot(label='Historical AQI', color='gray')
        prophet_forecast['yhat'].plot(label='Prophet Forecast', color='green')
        arima_forecast.plot(label='ARIMA Forecast (auto)', color='red')

        plt.title(f"AQI Forecast Comparison - {city_name}")
        plt.xlabel("Date")
        plt.ylabel("AQI")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"❌ auto_arima failed: {str(e)}")
'''
