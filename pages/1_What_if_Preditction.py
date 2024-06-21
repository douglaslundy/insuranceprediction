import pickle 
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Insurance Prediction')

#-- Title --#
st.sidebar.header("What if Preditction")
st.title('Insurance Prediction')

#-- Parameters --#
age = st.number_input(label='Age', value=18, min_value=18, max_value=120)

bmi = st.number_input(label='BMI', value=30.0)

children = st.slider(label='Children', min_value=0, max_value=5)
smoker = st.selectbox('Smoker', options=['no', 'yes'])


#-- Model --#

with open('../models/model.pkl', 'rb')  as model_file:
    model = pickle.load(model_file)

def prediction():
    df_input = pd.DataFrame([{
        'age': age,
        'bmi': bmi,
        'children': children,
        'smoker': smoker
    }])
    prediction = model.predict(df_input)[0]
    return prediction



if st.button('Predict'):
    try:
        insurance_cost = prediction()
        st.success(f'**Predicted insurance price:** ${insurance_cost:,.2f}')
    except Exception as error:
        st.error(f"Couldn'tpredict the input data. The following error occurred: \n\n{error} ")