import pandas as pd
import calendar


def load_and_clean_data(filepath):
    # Load data
    df = pd.read_csv(filepath)

    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Drop rows with missing Date, AQI or AQI_Bucket
    df.dropna(subset=['Date', 'AQI', 'AQI_Bucket'], inplace=True)

    # Define pollutant columns
    pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx',
                  'NH3', 'CO', 'SO2', 'O3', 'Benzene',
                  'Toluene', 'Xylene']

    # Fill missing pollutant values with median
    df[pollutants] = df[pollutants].fillna(df[pollutants].median())

    # Time-based features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Weekday'] = df['Date'].dt.weekday
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Month_Name'] = df['Month'].apply(lambda x: calendar.month_abbr[x])

    return df, pollutants
