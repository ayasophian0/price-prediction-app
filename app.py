import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import time

encoder = pickle.load(open('encoder', 'rb'))
scaler = pickle.load(open('scaler', 'rb'))
model = pickle.load(open('xgb_final_model', 'rb'))

st.write("""
         # Welcome to the Ultimate Car Price Predictor Machine!
         
         Let's play with the widgets in the slidebar 
         and then click on the button below to get the price of your dream car!
         """)

st.sidebar.header('This is where your inputs go!')

def user_input_features():
    make_model = st.sidebar.selectbox('Make and Model', ('SEAT Leon', 'Renault Megane', 'Ford Mustang', 'Hyundai i30',
       'Ford Focus', 'Peugeot 308', 'Opel Astra', 'Dacia Sandero',
       'Nissan Qashqai', 'Fiat 500', 'Ford Fiesta', 'Skoda Octavia',
       'Renault Clio', 'Fiat 500X', 'Fiat Tipo', 'Opel Corsa', 'SEAT Ibiza',
       'Dacia Duster', 'Opel Insignia', 'Peugeot 208'))
    body_type = st.sidebar.selectbox('Body Type', ('Sedan', 'Station wagon', 'Compact', 'Coupe', 'Off-Road/Pick-up',
       'Convertible'))
    Type = st.sidebar.selectbox('Type', ('Used', 'Pre-registered', 'Demonstration', "Employee's car"))
    mileage = st.sidebar.slider('Mileage (in km)', 0, 330000, 50000)
    Gearbox = st.sidebar.radio('Gearbox', ('Automatic', 'Manual', 'Semi-automatic'))
    fuel_type = st.sidebar.selectbox('Fuel', ('Benzine', 'Diesel', 'Other', 'Hybrid', 'LPG', 'Electric'))
    paint = st.sidebar.radio('Paint Type', ('Metallic', 'Uni/basic'))
    drivetrain = st.sidebar.radio('Drive Chain', ('Front', '4WD', 'Rear'))
    empty_weight = st.sidebar.slider('Empty Weight (in kg)', 500, 2500, 500)
    upholstery = st.sidebar.radio('Upholstery Type', ('Cloth', 'Leather', 'Alcantara', 'Other'))
    horsepower = st.sidebar.slider('Horsepower (hp)', 30, 600, 50)
    engine_size = st.sidebar.slider('Engine Size (in L)', 0.0, 5.0, 1.0)
    fuel_consumption = st.sidebar.slider('Fuel Consumption (in L/100km)', 0.0, 20.0, 5.0)
    age = st.sidebar.slider('Age (in years)', 0, 30, 5)

    data = {'make_model': make_model, 'body_type': body_type, 'type': Type, 'mileage': mileage,
            'gearbox': Gearbox, 'fuel_type': fuel_type, 'paint': paint,
            'drivetrain': drivetrain, 'empty_weight': empty_weight, 'upholstery': upholstery,
            'horsepower': horsepower, 'engine_size': engine_size,
            'fuel_consumption': fuel_consumption, 'age': age
            }
    
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
df.columns = df.columns.str.replace('_', ' ').str.title().str.strip()

file_ = open("aladin.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="the jinni gif">',
    unsafe_allow_html=True
)

st.markdown('###')

st.markdown('## Are you sure these are the features?')
st.dataframe(df.head(1), use_container_width=True)

encoded_df = encoder.transform(df)
scaled_df = scaler.transform(encoded_df)
prediction = model.predict(scaled_df)

st.markdown('###')

st.subheader('Now, you mortal, click the button below!')

if st.button('Me, me, me!'):
    time.sleep(1)
    st.write('### Hmmmmmm...')
    time.sleep(3)
    st.write('### Magic takes time...')
    time.sleep(2)
    st.success(f'## The price of your dream car is: €{round(prediction[0])}')
