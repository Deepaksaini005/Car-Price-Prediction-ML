import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# -----------------------
# Load Model
# -----------------------
model = joblib.load("car_price_model.pkl")
encoders = joblib.load("label_encoders.pkl")

# -----------------------
# CSS
# -----------------------

st.markdown("""
<style>

.main{
    background:#f5f7fa;
}

.title{
font-size:40px;
font-weight:bold;
text-align:center;
color:#0F62FE;
}

.sub{
text-align:center;
color:gray;
margin-bottom:30px;
}

.stButton>button{
width:100%;
background:#0F62FE;
color:white;
font-size:20px;
height:60px;
border-radius:12px;
border:none;
}

.stButton>button:hover{
background:#0043CE;
}

.metric{
background:white;
padding:20px;
border-radius:15px;
box-shadow:0px 0px 10px rgba(0,0,0,0.15);
}

</style>
""", unsafe_allow_html=True)

# -----------------------
# Header
# -----------------------

st.markdown("<div class='title'>🚗 Car Price Prediction</div>",unsafe_allow_html=True)

st.markdown("<div class='sub'>Predict the Resale Value of Your Car using Machine Learning</div>",unsafe_allow_html=True)

# -----------------------
# Sidebar
# -----------------------

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/744/744465.png", width=120)

st.sidebar.title("About")

st.sidebar.info("""
This application predicts the resale value of used cars.

Model Used:
Extra Trees Regressor

Made with ❤️ using Streamlit
""")

# -----------------------
# Inputs
# -----------------------

col1,col2=st.columns(2)

with col1:

    year=st.number_input("Manufacturing Year",2000,2026,2020)

    km_driven=st.number_input("Kilometers Driven",0,500000,50000)

    mileage=st.number_input("Mileage (km/l)",0.0,50.0,18.0)

    engine=st.number_input("Engine (CC)",500,6000,1200)

    max_power=st.number_input("Max Power",20.0,500.0,90.0)

with col2:

    seats=st.number_input("Seats",2,10,5)

    brand=st.selectbox(
        "Brand",
        encoders["brand"].classes_
    )

    fuel=st.selectbox(
        "Fuel",
        encoders["fuel"].classes_
    )

    seller_type=st.selectbox(
        "Seller Type",
        encoders["seller_type"].classes_
    )

    transmission=st.selectbox(
        "Transmission",
        encoders["transmission"].classes_
    )

    owner=st.selectbox(
        "Owner",
        encoders["owner"].classes_
    )

# -----------------------
# Encode
# -----------------------

brand=encoders["brand"].transform([brand])[0]

fuel=encoders["fuel"].transform([fuel])[0]

seller_type=encoders["seller_type"].transform([seller_type])[0]

transmission=encoders["transmission"].transform([transmission])[0]

owner=encoders["owner"].transform([owner])[0]

# -----------------------
# Prediction
# -----------------------

if st.button("🚀 Predict Car Price"):

    input_data=pd.DataFrame({

        "year":[year],
        "km_driven":[km_driven],
        "fuel":[fuel],
        "seller_type":[seller_type],
        "transmission":[transmission],
        "owner":[owner],
        "mileage":[mileage],
        "engine":[engine],
        "max_power":[max_power],
        "seats":[seats],
        "brand":[brand]

    })

    prediction=model.predict(input_data)[0]

    st.success("Prediction Completed")

    st.markdown("---")

    c1,c2=st.columns(2)

    with c1:

        st.metric(
            label="Estimated Price",
            value=f"₹ {prediction:,.0f}"
        )

    with c2:

        st.write("### Input Summary")

        st.write(f"**Brand :** {encoders['brand'].inverse_transform([brand])[0]}")
        st.write(f"**Fuel :** {encoders['fuel'].inverse_transform([fuel])[0]}")
        st.write(f"**Transmission :** {encoders['transmission'].inverse_transform([transmission])[0]}")
        st.write(f"**Mileage :** {mileage}")
        st.write(f"**Engine :** {engine} cc")
        st.write(f"**Power :** {max_power}")
        st.write(f"**Seats :** {seats}")

st.markdown("---")