import streamlit as st
import pandas as pd
from database import get_hospital_names, get_wards, check_data
from preprocessing import preprocess_ward_data
from prediction import process_single_ward, plot_forecast

# Page configuration
st.config.set_page_config(
    page_title="Horizon Health Forecasting",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Horizon Health - Hospital Ward Occupancy Forecasting System"}
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding-top: 0rem;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        font-size: 16px;
        font-weight: bold;
        padding: 10px 30px;
        border: none;
    }
    .stButton > button:hover {
        background-color: #0f4a7f;
    }
    </style>
""", unsafe_allow_html=True)

# Header with styling
st.markdown("# 🏥 Horizon Health - Hospital Ward Occupancy Forecasting")
st.markdown("### 📊 Intelligent Bed Prediction Dashboard")
st.markdown("---")

# Introduction section
with st.container():
    st.markdown("""
    Welcome to the **Horizon Health Bed Prediction Project Dashboard**. 
    
    This tool helps healthcare administrators forecast hospital bed occupancy across different wards,
    enabling better resource planning and patient care management.
    """)

st.markdown("---")

# Sidebar for input selection
st.sidebar.markdown("## ⚙️ Configuration")
st.sidebar.markdown("Select your hospital and ward details below:")

hospital_names = get_hospital_names()
selected_hospital = st.sidebar.selectbox("🏢 Select a Hospital", hospital_names)

wards = get_wards(selected_hospital)
selected_ward = st.sidebar.selectbox("🛏️ Select a Ward", wards)

st.sidebar.markdown("---")
st.sidebar.markdown("**Selected Options:**")
st.sidebar.write(f"**Hospital:** {selected_hospital}")
st.sidebar.write(f"**Ward:** {selected_ward}")

# Main content area
st.markdown("## 📈 Data Overview")

# Fetch and display ward data
ward_data = check_data(selected_hospital, selected_ward)

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("### Ward Data Preview")
    st.dataframe(ward_data.head(), use_container_width=True)

with col2:
    st.markdown("### 📋 Data Summary")
    st.metric("Total Records", len(ward_data))
    st.metric("Columns", len(ward_data.columns))

# Process data
ts_data = preprocess_ward_data(ward_data)

# Forecast section
st.markdown("---")
st.markdown("## 🔮 Generate Forecast")
st.write("Click the button below to generate an occupancy forecast for the selected ward.")

if st.button("📊 Generate Forecast", use_container_width=True):
    with st.spinner("🔄 Processing forecast... This may take a moment..."):
        forecast_df, results_dict, train, test = process_single_ward(ts_data, selected_hospital, selected_ward)
        fig = plot_forecast(ts_data, train, test, forecast_df, results_dict, selected_hospital, selected_ward)
        
    st.markdown("### 📈 Forecast Results")
    st.pyplot(fig)
    st.success("✅ Forecast generated successfully!")