import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px  # Yêu cầu có trong requirements.txt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- UI Title ---
st.title("Demand Forecasting Dashboard")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("sales_data.csv")  # Bạn có thể đổi tên file tại đây
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# --- Sidebar Filters ---
st.sidebar.header("Filter Options")
product_list = df["Product Category"].unique().tolist()
selected_product = st.sidebar.selectbox("Select Product Category", product_list)

date_range = st.sidebar.date_input("Select Date Range",
    [df["Date"].min(), df["Date"].max()])

# --- Filtered Data ---
filtered_df = df[
    (df["Product Category"] == selected_product) &
    (df["Date"] >= pd.to_datetime(date_range[0])) &
    (df["Date"] <= pd.to_datetime(date_range[1]))
]

# --- Group by Date ---
grouped_df = filtered_df.groupby("Date")["Quantity"].sum().reset_index()

# --- Show Data ---
st.subheader(f"Total Quantity Sold Over Time: {selected_product}")
fig = px.line(grouped_df, x="Date", y="Quantity", title="Sales Over Time")
st.plotly_chart(fig)

# --- Model Training ---
st.subheader("Train Forecasting Model")
grouped_df['day'] = np.arange(len(grouped_df)).reshape(-1, 1)

X = grouped_df[['day']]
y = grouped_df['Quantity']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# --- Model Evaluation ---
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write("**Model Evaluation Metrics**")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R² Score: {r2:.2f}")

# --- Forecast Visualization ---
future_days = st.slider("Forecast Days into Future", 7, 60, 30)
future_X = np.arange(len(grouped_df), len(grouped_df) + future_days).reshape(-1, 1)
future_y = model.predict(future_X)

future_df = pd.DataFrame({
    "Date": pd.date_range(start=grouped_df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=future_days),
    "Predicted Quantity": future_y
})

combined_df = pd.concat([
    grouped_df[["Date", "Quantity"]].rename(columns={"Quantity": "Demand"}),
    future_df.rename(columns={"Predicted Quantity": "Demand"})
])

combined_df["Type"] = ["Historical"] * len(grouped_df) + ["Forecast"] * len(future_df)

st.subheader("Forecast Result")
fig2 = px.line(combined_df, x="Date", y="Demand", color="Type", title="Demand Forecast")
st.plotly_chart(fig2)
