import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

model = joblib.load("model/car_price_model.pkl")
make_encoder = joblib.load("model/make_encoder.pkl")
transmission_encoder = joblib.load("model/transmission_encoder.pkl")

st.title("Car Price Estimator")

st.markdown("Enter your car details to estimate its resale price.")

brand = st.selectbox("Select car brand", sorted(make_encoder.classes_))
year = st.number_input("Enter year of manufacture", min_value=1990, max_value=2025, step=1)
transmission = st.selectbox("Select transmission type", sorted(transmission_encoder.classes_))
miles = st.number_input("Enter miles driven", min_value=0, step=1000)

if st.button("Estimate Price"):
    try:
        make_enc = make_encoder.transform([brand])[0]
        transmission_enc = transmission_encoder.transform([transmission])[0]

        features = [[year, miles, make_enc, transmission_enc]]
        price = model.predict(features)[0]

        error_margin = 2000
        lower = int(price - error_margin)
        upper = int(price + error_margin)

        st.success(f"Estimated Price: ${int(price):,} ± ${error_margin:,}")
        st.markdown(f"Price Range: **${lower:,} - ${upper:,}**")

        row = {
            'brand': brand,
            'year': year,
            'transmission': transmission,
            'miles_driven': miles,
            'predicted_price': int(price)
        }

        log_file = "prediction_log.csv"
        df_row = pd.DataFrame([row])
        if os.path.exists(log_file):
            df_row.to_csv(log_file, mode='a', header=False, index=False)
        else:
            df_row.to_csv(log_file, index=False)

    except Exception as e:
        st.error(f"Something went wrong: {e}")


        st.markdown("___")
        st.markdown("Built by **Sriniketh**", unsafe_allow_html=True)
