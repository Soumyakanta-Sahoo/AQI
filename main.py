from src.data_preprocessing import load_and_clean_data
from src.eda import (
    plot_aqi_distribution,
    plot_aqi_category_counts,
    plot_pollutant_correlation,
    plot_monthly_aqi_trend,
    plot_citywise_trends
)
from src.models import (
    prepare_features,
    train_regression_model,
    train_classification_model,
    plot_feature_importance,
    export_classification_results
)
from src.forecast import (
    prepare_city_timeseries,
    forecast_with_prophet,
    decompose_city_aqi,
    # forecast_with_arima,
    # compare_forecasts
)


def main():
    # ------------------ Load Data ------------------
    data_path = "data/city_day.csv"
    df, pollutants = load_and_clean_data(data_path)

    print("✅ Data Loaded and Cleaned")
    print(f"Shape: {df.shape}")
    print(f"Pollutants: {pollutants}")

    # ---------------------- EDA ----------------------
    print("\n🔍 Running EDA...")
    plot_aqi_distribution(df)
    plot_aqi_category_counts(df)
    plot_pollutant_correlation(df, pollutants)
    plot_monthly_aqi_trend(df)
    plot_citywise_trends(df)

    # ------------------- Feature Prep -------------------
    print("\n⚙️ Preparing Features...")
    X, y_reg, y_clf, le = prepare_features(df, pollutants)

    # ------------------- Modeling -------------------
    print("\n🚀 Training Models...")
    reg_model = train_regression_model(X, y_reg)
    clf_model = train_classification_model(X, y_clf, le)

    # ------------------- Extras -------------------
    print("\n📊 Feature Importance:")
    plot_feature_importance(reg_model, X.columns)

    print("\n💾 Exporting classification predictions...")
    export_classification_results(X, y_clf, clf_model, le)

    # ------------------ Time Series Forecasting ------------------
    print("\n📅 Time Series Forecasting...")
    city_ts = prepare_city_timeseries(df)
    city_name = "Delhi"  # 👈 You can change this to any valid city

    forecast_with_prophet(city_ts, city_name)
    decompose_city_aqi(city_ts, city_name)

    # Optional: ARIMA & comparison (disabled)
    # forecast_with_arima(city_ts, city_name)
    # compare_forecasts(city_ts, city_name)


if __name__ == "__main__":
    main()
