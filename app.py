import streamlit as st
from backend import ModelPredictor, normalize_data
import pandas as pd

def predict(normalized_df):
    """Make a prediction using the normalized dataframe."""
    model = ModelPredictor('xgb_model.json')
    result = model.predict(normalized_df)
    return result 

# Options for selectbox widgets
family_status_options = ['Civil marriage', 'Married', 'Separated', 'Single / not married']
education_level_options = ['Higher education', 'Incomplete higher', 'Lower secondary','Secondary / secondary special']

st.title('Default Prediction')

# Create selectbox and number input widgets for user input
NAME_FAMILY_STATUS = st.selectbox('Family status', family_status_options, key='NAME_FAMILY_STATUS')
NAME_EDUCATION_TYPE = st.selectbox('Education level', education_level_options, key='NAME_EDUCATION_TYPE')
AMT_INCOME_TOTAL = st.number_input('Total income', 0, 10000000, 100000, key='AMT_INCOME_TOTAL')
AMT_CREDIT = st.number_input('Credit amount', 0, 10000000, 100000, key='AMT_CREDIT')
AMT_ANNUITY = st.number_input('Annuity amount', 0, 300000, 50000 , key='AMT_ANNUITY')
EXT_SOURCE_2 = st.slider('Normalized credit score', 0., 1., 0.8, step=0.01, key='EXT_SOURCE_2')
AGE = st.slider('Age', 18, 80, 25, key='AGE')
CNT_APPLICATIONS = st.slider('Number of previous applications', 0, 100, key='CNT_APPLICATIONS')

# Make a prediction when the 'Predict' button is clicked
if st.button('Predict'):
     # Create a dataframe with the user inputs
    input_df = pd.DataFrame({
        'AMT_INCOME_TOTAL': [AMT_INCOME_TOTAL], 
        'AMT_CREDIT': [AMT_CREDIT], 
        'AMT_ANNUITY': [AMT_ANNUITY], 
        'AGE': [AGE], 
        'CNT_APPLICATIONS': [CNT_APPLICATIONS],
        'EXT_SOURCE_2': [EXT_SOURCE_2], 
        'NAME_FAMILY_STATUS': NAME_FAMILY_STATUS, 
        'NAME_EDUCATION_TYPE': NAME_EDUCATION_TYPE
    })
    # Normalize the input data
    normalized_df = normalize_data(input_df)

    # Make a prediction using the normalized data
    result = predict(normalized_df)

    # Display the prediction result
    if result < 0.5:
        st.success('Default cannot be predicted')
    else:
        st.error('Default can be predicted')
