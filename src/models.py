from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def prepare_features(df, pollutants):
    # Label encode AQI_Bucket
    le = LabelEncoder()
    df['AQI_Label'] = le.fit_transform(df['AQI_Bucket'])

    # Features
    features = pollutants + ['Year', 'Month', 'Day', 'Weekday']
    X = df[features]
    y_reg = df['AQI']
    y_clf = df['AQI_Label']

    return X, y_reg, y_clf, le


def train_regression_model(X, y):
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Regressors
    regressors = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(),
        "AdaBoost": AdaBoostRegressor(),
        "XGBoost": XGBRegressor(objective="reg:squarederror", verbosity=0)
    }

    print("📈 Regression Results:")
    best_model = None
    best_r2 = -np.inf

    for name, model in regressors.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        print(f"\n🔹 {name}")
        print(f" MAE:  {mae:.2f}")
        print(f" RMSE: {rmse:.2f}")
        print(f" R²:   {r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model = model

    return best_model


def train_classification_model(X, y, label_encoder):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    print("\n📊 Classification Results:")
    print(f" Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(" Classification Report:\n", classification_report(y_test, preds, target_names=label_encoder.classes_))

    return model


def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()

        plt.figure(figsize=(10, 6))
        importances.plot(kind='barh', color='steelblue')
        plt.title("🔍 Feature Importances (Random Forest)")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ Model has no feature_importances_ attribute.")


def export_classification_results(X, y_true, model, label_encoder, filename="output/classification_results.csv"):
    preds = model.predict(X)
    df_results = pd.DataFrame(X, columns=X.columns)
    df_results['Actual_Label'] = label_encoder.inverse_transform(y_true)
    df_results['Predicted_Label'] = label_encoder.inverse_transform(preds)

    df_results.to_csv(filename, index=False)
    print(f"📁 Classification results saved to: {filename}")
