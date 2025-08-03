import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

st.set_page_config(layout="wide")
st.title("Demand Forecasting Dashboard")

@st.cache_data
def load_and_preprocess_data():
    # Đọc dữ liệu gốc
    df = pd.read_csv("data.csv")

    # Chuyển đổi cột Date thành datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)

    # Mã hóa các cột phân loại
    le_dict = {}
    for col in ['Gender', 'Product Category']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    # Feature Engineering
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Weekday'] = df['Date'].dt.weekday

    return df, le_dict

df, le_dict = load_and_preprocess_data()

# Chia dữ liệu train/test
features = ['Gender', 'Age', 'Product Category', 'Quantity', 'Price per Unit', 'Year', 'Month', 'Day', 'Weekday']
X = df[features]
y = df['Total Amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Dashboard
st.sidebar.header("Filter Options")
selected_category = st.sidebar.selectbox("Select Product Category", options=sorted(df['Product Category'].unique()))
selected_year = st.sidebar.selectbox("Select Year", options=sorted(df['Year'].unique()))

filtered_df = df[(df['Product Category'] == selected_category) & (df['Year'] == selected_year)]

# Biểu đồ dự đoán
st.subheader("Actual vs Predicted Total Amount")
fig1 = px.scatter(
    x=y_test,
    y=y_pred,
    labels={'x': 'Actual Amount', 'y': 'Predicted Amount'},
    title="Actual vs Predicted",
    opacity=0.7
)
st.plotly_chart(fig1, use_container_width=True)

# Biểu đồ theo thời gian
st.subheader("Total Amount by Date")
daily_total = filtered_df.groupby('Date')['Total Amount'].sum().reset_index()
fig2 = px.line(daily_total, x='Date', y='Total Amount', title='Daily Sales Amount')
st.plotly_chart(fig2, use_container_width=True)

# Hiển thị dữ liệu đã lọc
st.subheader("Filtered Data Sample")
st.dataframe(filtered_df.head(20))

# Đánh giá mô hình
st.subheader("Model Evaluation")
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"**R-squared Score:** {r2:.2f}")

# Kết luận và cải tiến
st.markdown("""
### 📌 Kết luận và Đề xuất cải tiến:
- Mô hình Linear Regression đã học được mối quan hệ tổng thể giữa các đặc trưng và Total Amount.
- **R-squared** cho thấy mức độ phù hợp tương đối, nhưng có thể được cải thiện bằng:
    - Thêm các biến tương tác hoặc biến thời gian (lag, moving average).
    - Sử dụng mô hình phi tuyến như Random Forest hoặc XGBoost.
    - Huấn luyện riêng cho từng Product Category.
""")
