import matplotlib.pyplot as plt
import seaborn as sns
import calendar


def plot_aqi_distribution(df, save_path="output/aqi_distribution.png"):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['AQI'], bins=50, kde=True, color='darkorange')
    plt.title("Distribution of AQI Values")
    plt.xlabel("AQI")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"🖼️ Saved: {save_path}")
    plt.show()


def plot_aqi_category_counts(df):
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='AQI_Bucket', order=df['AQI_Bucket'].value_counts().index, hue='AQI_Bucket', palette='coolwarm', legend=False)
    plt.title("AQI Category Distribution")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_pollutant_correlation(df, pollutants):
    plt.figure(figsize=(14, 10))
    sns.heatmap(df[pollutants + ['AQI']].corr(), annot=True, fmt=".2f", cmap='RdYlBu_r')
    plt.title("Correlation Heatmap: Pollutants vs AQI")
    plt.tight_layout()
    plt.show()


def plot_monthly_aqi_trend(df):
    monthly_avg = df.groupby(['Year', 'Month_Name'])['AQI'].mean().reset_index()
    monthly_avg['Month_Num'] = monthly_avg['Month_Name'].apply(lambda x: list(calendar.month_abbr).index(x))
    monthly_avg = monthly_avg.sort_values(['Year', 'Month_Num'])

    monthly_avg['YM'] = monthly_avg['Year'].astype(str) + '-' + monthly_avg['Month_Name']
    plt.figure(figsize=(16, 5))
    sns.lineplot(data=monthly_avg, x='YM', y='AQI', marker='o', color='green')
    plt.xticks(rotation=45)
    plt.title("Average AQI Over Time (Monthly)")
    plt.ylabel("AQI")
    plt.tight_layout()
    plt.show()


def plot_citywise_trends(df, top_n=6):
    top_cities = df['City'].value_counts().nlargest(top_n).index
    plt.figure(figsize=(14, 8))

    for city in top_cities:
        subset = df[df['City'] == city].groupby('Date')['AQI'].mean().reset_index()
        subset['AQI_Smoothed'] = subset['AQI'].rolling(window=7, min_periods=1).mean()  # 7-day rolling average
        plt.plot(subset['Date'], subset['AQI_Smoothed'], label=city)

    plt.title(f"Smoothed AQI Trends for Top {top_n} Cities")
    plt.xlabel("Date")
    plt.ylabel("AQI (7-day MA)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_city_aqi_distribution(df, city_name):
    city_df = df[df['City'] == city_name]
    plt.figure(figsize=(10, 5))
    sns.histplot(city_df['AQI'], bins=40, kde=True, color='purple')
    plt.title(f"AQI Distribution for {city_name}")
    plt.xlabel("AQI")
    plt.tight_layout()
    plt.show()
