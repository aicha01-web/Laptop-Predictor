import logging
import streamlit as st
import pickle
import numpy as np

_logger = logging.getLogger(__name__)

# import the model
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('data.pkl', 'rb'))

st.title("Laptop Price Predictor")

st.markdown("Just fill out the specifications you need for your laptop and hit the button, and it's done! You will get the estimated price for the laptop.")

# brand
company = st.selectbox('Brand', df['Company'].unique())

# type of laptop
type = st.selectbox('Type', df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)', df['Ram'].unique())

# weight
weight = st.selectbox('Weight (in KG)', df['Weight'].unique())

#flash storage
flash_storage = st.number_input("Flash storage")

#hybrid
hybrid = st.selectbox("Hybride", ["No", "Yes"])

# screen size
screen_size = st.selectbox('Screen Size (in Inches)', df['Inches'].unique())

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600','2560x1440', '2304x1440'])


# cpu
cpu = st.selectbox('CPU', df['Cpu brand'].unique())

hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])

ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])

gpu = st.selectbox('GPU Brand', df['Gpu'].unique())

os = st.selectbox('OS', df['OpSys'].unique())

if st.button('Predict Price'):
    # query
    ppi=None
    if hybrid == 'Yes':
        hybrid = 1
    else:
        hybrid = 0


    X_res = int(resolution.split('x')[0])
    _logger.log(1,X_res)
    Y_res = int(resolution.split('x')[1])
    memory = hdd + ssd + flash_storage
    resolution = X_res * Y_res
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size
    query = np.array([company, type, screen_size, resolution, ram, memory ,gpu, os, weight,cpu, hdd, ssd, hybrid,flash_storage, X_res,Y_res, ppi])
    query = query.reshape(1, 17)
    st.balloons()
    # st.title(pipe.predict(query))
    st.title("The predicted price for this configuration is " + str(np.floor(pipe.predict(query)[0]))+ ' FCFA')