# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(layout="wide")

st.title("📊 Product Demand Forecasting - ABC Manufacturing")
st.markdown("This app trains a simple ML model to predict **Total Amount** using historical data.")

# --- Load and preprocess data ---
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("processed_data.csv")
    df.dropna(inplace=True)

    # Convert Date column
    df['Date'] = pd.to_datetime(df['Date'])

    # Encode categorical variables
    cat_cols = ['Customer Gender', 'Product Category', 'Payment Method']
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    return df, le_dict

df, le_dict = load_and_preprocess_data()

# --- Train model ---
X = df[['Quantity', 'Price per Unit', 'Age', 'Customer Gender', 'Product Category', 'Payment Method']]
y = df['Total Amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# --- Predict and attach ---
df['Predicted Amount'] = model.predict(X)

# --- Sidebar filters ---
st.sidebar.header("Filter Options")
product_names = le_dict['Product Category'].classes_
selected_category = st.sidebar.selectbox("Product Category", product_names)

# Map back to encoded value
selected_category_code = le_dict['Product Category'].transform([selected_category])[0]

# Filter data
filtered_df = df[df['Product Category'] == selected_category_code].sort_values("Date")

# --- Plot ---
st.subheader(f"📈 Total Amount: Actual vs Predicted for **{selected_category}**")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(filtered_df['Date'], filtered_df['Total Amount'], label='Actual', color='blue')
ax.plot(filtered_df['Date'], filtered_df['Predicted Amount'], label='Predicted', color='orange')
ax.fill_between(filtered_df['Date'],
                filtered_df['Predicted Amount'] * 0.95,
                filtered_df['Predicted Amount'] * 1.05,
                color='orange', alpha=0.2, label='Confidence Interval (±5%)')
ax.set_xlabel("Date")
ax.set_ylabel("Total Amount")
ax.legend()
ax.grid(True)
st.pyplot(fig)
