import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Insurance Prediction', page_icon='img/stethoscope.png')

st.sidebar.header("Insurance prediction")
st.title("Insurance prediction")

st.markdown("Preidct medical insurance based using a csv file:")

#-- Model --#
with open('models/model.pkl', 'rb')  as model_file:
    model = pickle.load(model_file)

data = st.file_uploader('Upload your file')

if data:
    df_input = pd.read_csv(data)
    insurance_prediction = model.predict(df_input)
    df_output = df_input.assign(prediction=insurance_prediction)

    st.markdown('Insurance cost prediction')
    st.write(df_output)
    st.download_button(label='Download CSV', data=df_output.to_csv(index=False).encode('utf-8'), 
                       mime='text/csv', file_name='prediction_insurance_coast')
