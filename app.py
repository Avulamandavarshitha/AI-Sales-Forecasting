import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from math import ceil

st.set_page_config(
    page_title="AI Sales Forecasting",
    page_icon="📊",
    layout="wide"
)

st.title("AI-Based Sales Forecasting and Intelligent Inventory Optimization")
st.write("Sales prediction and inventory management using Machine Learning.")

# Load dataset
data = pd.read_csv("sales_data.csv")
data["Date"] = pd.to_datetime(data["Date"])

st.header("1. Sales Dataset")
st.dataframe(data, use_container_width=True)

# Feature engineering
data["Month"] = data["Date"].dt.month
data["Day"] = data["Date"].dt.day
data["DayOfWeek"] = data["Date"].dt.dayofweek
data["DayOfYear"] = data["Date"].dt.dayofyear

features = ["Month", "Day", "DayOfWeek", "DayOfYear", "Price"]

stock = {
    "Laptop": 20,
    "Mouse": 50,
    "Keyboard": 10
}

results = []
forecast_details = []

st.header("2. Sales Forecasting")

for product in data["Product"].unique():

    product_data = data[data["Product"] == product].copy()
    product_data = product_data.sort_values("Date")

    X = product_data[features]
    y = product_data["Units_Sold"]

    # Chronological 80:20 split
    split = int(len(product_data) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Model evaluation
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    # Train final model using all available data
    model.fit(X, y)

    # Predict next 7 days
    last_date = product_data["Date"].max()

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=7
    )

    future = pd.DataFrame({
        "Date": future_dates,
        "Price": product_data["Price"].iloc[0]
    })

    future["Month"] = future["Date"].dt.month
    future["Day"] = future["Date"].dt.day
    future["DayOfWeek"] = future["Date"].dt.dayofweek
    future["DayOfYear"] = future["Date"].dt.dayofyear

    future_predictions = model.predict(future[features])
    future_predictions = np.maximum(future_predictions, 0)

    forecast_demand = future_predictions.sum()

    # Inventory calculation
    safety_stock = ceil(forecast_demand * 0.10)
    current_stock = stock[product]

    suggested_order = max(
        0,
        ceil(forecast_demand) + safety_stock - current_stock
    )

    results.append({
        "Product": product,
        "MAE": round(mae, 2),
        "Forecast Demand (7 Days)": round(forecast_demand, 2),
        "Current Stock": current_stock,
        "Safety Stock": safety_stock,
        "Suggested Order": suggested_order
    })

    for date, prediction in zip(future_dates, future_predictions):
        forecast_details.append({
            "Date": date,
            "Product": product,
            "Predicted Units": round(float(prediction), 2)
        })

    st.subheader(f"{product} Forecast")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")

    fig, ax = plt.subplots()
    ax.plot(
        product_data["Date"],
        product_data["Units_Sold"],
        marker="o",
        label="Actual Sales"
    )
    ax.plot(
        future_dates,
        future_predictions,
        marker="o",
        linestyle="--",
        label="Predicted Sales"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Units Sold")
    ax.set_title(f"{product} Sales Forecast")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Results table
results_df = pd.DataFrame(results)
forecast_df = pd.DataFrame(forecast_details)

st.header("3. Inventory Optimization")
st.dataframe(results_df, use_container_width=True)

st.header("4. Seven-Day Demand Forecast")
st.dataframe(forecast_df, use_container_width=True)

# Download results
st.download_button(
    "Download Forecast Results",
    forecast_df.to_csv(index=False),
    file_name="forecast_results.csv",
    mime="text/csv"
)

st.success("Sales forecasting and inventory analysis completed!")
