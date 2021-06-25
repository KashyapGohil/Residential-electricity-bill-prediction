import pickle
import re
import pandas as pd
import streamlit as st
import xgboost
import lightgbm
from exceptions import ModelNotValidException
import warnings
warnings.filterwarnings('ignore')


# title_temp="""
# <div style="background-color:tomato;padding:10px">
# <h2 style="color:white;text-align:center;"> Name/h2>
# </div>
# """ 
# st.markdown(title_temp, unsafe_allow_html = True)

st.title("Residential Electricity Bill Prediction")
st.header("Predicts the next month’s electricity bill amount for a given household")

lightgbm_optuna_file = 'models/LightGBM_Optuna.pkl'
xgboost_optuna_file = 'models/xgb_Optuna.pkl'


def model_loading():
    lgbm_optuna_model = pickle.load(open(lightgbm_optuna_file,'rb'))
    xgb_optuna_model = pickle.load(open(xgboost_optuna_file,'rb'))
    return lgbm_optuna_model,xgb_optuna_model   


model_load_state = st.text('Loading Model...')
lgbm_optuna_model,xgb_optuna_model = model_loading()
model_load_state.text('Loading Model...done!')

choose_model = st.sidebar.selectbox(
    "Pick model you'd like to use",
    ("LightGBM + Optuna",
     "XGBoost + Optuna",
     ) 
)

if choose_model == "LightGBM + Optuna":
    model=lgbm_optuna_model
else:
    model=xgb_optuna_model



## defining the function which will make the prediction using the data which the user inputs
def prediction(model,DOLELWTH, CDD30YR, HDD30YR, HEATROOM, NCOMBATH, TOTROOMS,
               TOTSQFT, DOLELCOL, BEDROOMS, KWH):

    #st.write(f'model',model )
    ## XGBoost + Optuna
    if isinstance(model, xgboost.core.Booster): 
        x = pd.DataFrame(pd.Series([HDD30YR,CDD30YR,BEDROOMS,NCOMBATH,TOTROOMS,TOTSQFT,HEATROOM,
        KWH,DOLELCOL,DOLELWTH],index=['HDD30YR','CDD30YR','BEDROOMS','NCOMBATH','TOTROOMS','TOTSQFT','HEATROOM',
        'KWH','DOLELCOL','DOLELWTH'])).transpose()
        y=[]
        dmatrix = xgboost.DMatrix(x,y)
        prediction =model.predict(dmatrix)
        return prediction

    ## LightGBM + Optuna    
    elif isinstance(model, lightgbm.basic.Booster):
        prediction = model.predict([[HDD30YR,CDD30YR,BEDROOMS,NCOMBATH,TOTROOMS,TOTSQFT,HEATROOM,
        KWH,DOLELCOL,DOLELWTH]])
        return prediction

    else:
        raise ModelNotValidException('Please select a valid model.')

#this is main function define webpage
def main():
    
    # following lines create boxes in which user can enter data required to make prediction 
    
    BEDROOMS=st.sidebar.slider('Number of Bed Rooms',1,13)
    NCOMBATH=st.sidebar.slider('Number of full bathrooms',1,8)
    TOTROOMS=st.sidebar.slider('Total number of rooms',1,23)
    HEATROOM=st.sidebar.slider('Number of rooms heated',1,23)
    TOTSQFT=st.number_input('Total square footage',min_value=100,max_value=16000)
    st.write('Range : (100-16000) square feet' )
    HDD30YR=st.number_input('Heating degree days',min_value=0,max_value=13000)
    st.write('Range : (0-13000) HDD' )
    CDD30YR=st.number_input('Cooling degree days',min_value=0,max_value=5000)
    st.write('Range : (0-5000) CDD' )
    KWH=st.number_input('Total Site Electricity usage',min_value=17,max_value=150000)
    st.write('Range : (17-150000) kilowatt-hours' )
    DOLELCOL=st.number_input('Electricity cost for air-conditioning, central and window/wall',min_value=0,max_value=7000)
    st.write('Range : (0-7000) dollars($)' )
    DOLELWTH=st.number_input('Electricity cost for water heating',min_value=0,max_value=2000)
    st.write('Range : (0-2000) dollars($)' )

    result=""
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"):
        try:
            result = prediction(model,DOLELWTH, CDD30YR, HDD30YR, HEATROOM, NCOMBATH, TOTROOMS, TOTSQFT,
                                     DOLELCOL, BEDROOMS, KWH)
            st.success(f"Your Total Electricity Cost is ${result[0] :.2f}.") 
            st.write('Done !!')
        except ModelNotValidException as e:
            st.write(f'Error : {e}')

     
if __name__=='__main__': 
    main()



