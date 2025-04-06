import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and dataset
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Set page configuration
st.set_page_config(page_title="Laptop Price Predictor", layout="centered")

# Light Theme Styling
st.markdown("""
    <style>
        body {
            background-color: #f2f2f2;
            color: #000000;
        }
        .title {
            text-align: center;
            font-size: 36px;
            color: #222222;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .description {
            text-align: center;
            font-size: 18px;
            color: #444444;
            margin-bottom: 25px;
            font-weight: 500;
        }
        label, .stSelectbox label, .stSlider label, .stTextInput label, .stNumberInput label {
            color: #111111 !important;
            font-weight: bold !important;
        }
    </style>
""", unsafe_allow_html=True)

# Title and Description
st.markdown('<div class="title">Laptop Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="description">Enter your desired specifications to get an idea of your laptop price</div>', unsafe_allow_html=True)

# Input Fields
brand = st.selectbox('Brand', df['Company'].unique())
type_ = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM (in GB)', sorted(df['Ram'].unique()))
weight = st.number_input('Weight of the laptop')

touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS Display', ['No', 'Yes'])

screen_size = st.number_input('Screen Size (in inches)')

resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900',
                                                 '3840x2160', '3200x1800', '2880x1800',
                                                 '2560x1600', '2560x1440', '2304x1440'])

cpu = st.selectbox('CPU', df['Cpu brand'].unique())
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (in GB)', [0, 128, 256, 512, 1024])
gpu = st.selectbox('GPU', df['Gpu brand'].unique())
os = st.selectbox('OS', df['os'].unique())

# Prediction Button
if st.button('Predict Price'):
    try:
        x_res = int(resolution.split('x')[0])
        y_res = int(resolution.split('x')[1])
        ppi = ((x_res**2 + y_res**2) ** 0.5) / screen_size

        touchscreen = 1 if touchscreen == 'Yes' else 0
        ips = 1 if ips == 'Yes' else 0

        query = pd.DataFrame([{
            'Company': brand,
            'TypeName': type_,
            'Ram': ram,
            'Weight': weight,
            'Touchscreen': touchscreen,
            'Ips': ips,
            'ppi': ppi,
            'Cpu brand': cpu,
            'HDD': hdd,
            'SSD': ssd,
            'Gpu brand': gpu,
            'os': os
        }])

        predicted_price = np.exp(pipe.predict(query)[0])
        st.success(f"The predicted price of this configuration is ₹ {int(predicted_price)}")

    except ZeroDivisionError:
        st.error("Screen size cannot be zero. Please enter a valid value.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
