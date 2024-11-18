import pandas as pd                  # Pandas
import numpy as np                   # Numpy
from matplotlib import pyplot as plt # Matplotlib

# Package to implement ML Algorithms
import sklearn
from sklearn.ensemble import RandomForestRegressor # Random Forest

# Import MAPIE to calculate prediction intervals
from mapie.regression import MapieRegressor

# To calculate coverage score
from mapie.metrics import regression_coverage_score

# Package for data partitioning
from sklearn.model_selection import train_test_split

# Package to record time
import time

# Module to save and load Python objects to and from files
import pickle 

# Ignore Deprecation Warnings
import warnings
warnings.filterwarnings('ignore')
import streamlit as st

rf_pickle = open('random_forest_FHealth.pickle', 'rb') 
rf_model = pickle.load(rf_pickle)
rf_pickle.close() 

dt_pickle = open('decision_tree_FHealth.pickle', 'rb') 
dt_model = pickle.load(dt_pickle)
dt_pickle.close() 

ada_pickle = open('ada_FHealth.pickle', 'rb') 
ada_model = pickle.load(ada_pickle)
ada_pickle.close() 

sv_pickle = open('sv_FHealth.pickle', 'rb') 
sv_model = pickle.load(sv_pickle) 
sv_pickle.close() 



df = pd.read_csv('fetal_health.csv')

sample = df.head(5)



with st.sidebar:
    st.write("Fetal Health Features Input")
    input = st.file_uploader("Upload your data")
    st.warning('Ensure your data strictly follows the format outlined below', icon="⚠️")
    st.write(sample)
    model = st.radio("Choose Model for Prediction", ["Random Forest", "Decision Tree", "AdaBoost", "Soft Voting"])
    st.write("You selected " + model)
    
    
st.title("Fetal Health Classification: A Machine Learning App")
st.image('fetal_health_image.gif', width = 650)
st.write("Utilize our advanced Machine Learning application to predict fetal health applications")




if input is not None:
    inputdf = pd.read_csv(input)
    inputdf2 = inputdf.copy()
    X = df.drop(columns = ['fetal_health'])
    y = df['fetal_health']
    X_encoded = pd.get_dummies(X)
    if model == 'Random Forest':
        inputdf['Predicted Fetal Health'] = rf_model.predict(inputdf) 
    elif model == 'Decision Tree':
        inputdf['Predicted Fetal Health'] = dt_model.predict(inputdf)
    elif model == 'AdaBoost':
        inputdf['Predicted Fetal Health'] = ada_model.predict(inputdf)
    else:
        inputdf['Predicted Fetal Health'] = sv_model.predict(inputdf)
    inputdf.loc[inputdf['Predicted Fetal Health'] == 1, 'Predicted Fetal Health'] = 'Normal'
    inputdf.loc[inputdf['Predicted Fetal Health'] == 2, 'Predicted Fetal Health'] = 'Suspect'
    inputdf.loc[inputdf['Predicted Fetal Health'] == 3, 'Predicted Fetal Health'] = 'Pathological'
    if model == 'Random Forest':
        inputdf['Prediction Probability'] = rf_model.predict_proba(inputdf2).max(axis=1)
    elif model == 'Decision Tree':
        inputdf['Prediction Probability'] = dt_model.predict_proba(inputdf2).max(axis=1)
    elif model == 'AdaBoost':
        inputdf['Prediction Probability'] = ada_model.predict_proba(inputdf2).max(axis=1)
    else:
        inputdf['Prediction Probability'] = sv_model.predict_proba(inputdf2).max(axis=1)
    
    def highlight_SLA(series):
        lime = 'background-color: lime'
        yellow = 'background-color: yellow'
        orange = 'background-color: orange'
        return [lime if value == 'Normal' else yellow if value == 'Suspect' else orange for value in series]
    slice_SLA = ['Predicted Fetal Health']
    inputdf = inputdf.style.apply(highlight_SLA, subset = slice_SLA)
    st.write(inputdf)
else:
    st.markdown('*Please upload data to proceed*')






if input is not None:
    st.subheader("Model Performance and Insights")
    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Classification Report", "Feature Importance"])

# Tab 1: Visualizing Confusion Matrix
    with tab1:
        st.write("### Confusion Matrix")
        if model == 'Random Forest':
            st.image('FH_RF_CM.svg')
        elif model == 'Decision Tree':
            st.image('FH_DT_CM.svg')
        elif model == 'AdaBoost':
            st.image('FH_ADA_CM.svg')
        else:
            st.image('FH_SV_CM.svg')
        st.caption("Confusion Matrix of model predictions.")

# Tab 2: Classification Report
    with tab2:
        st.write("### Classification Report")
        if model == 'Random Forest':
            report_df = pd.read_csv('FH_RF_CR.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
        elif model == 'Decision Tree':
            report_df = pd.read_csv('FH_DT_CR.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
        elif model == 'AdaBoost':
            report_df = pd.read_csv('FH_ADA_CR.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
        else:
            report_df = pd.read_csv('FH_SV_CR.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
        st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each fetal health.")

# Tab 3: Feature Importance
    with tab3:
        st.write("### Feature Importance")
        if model == 'Random Forest':
            st.image('FH_RF_FI.svg')
        elif model == 'Decision Tree':
            st.image('FH_DT_FI.svg')
        elif model == 'AdaBoost':
            st.image('FH_ADA_FI.svg')
        else:
            st.image('FH_SV_FI.svg')
        st.caption("Features used in this prediction are ranked by relative importance.")
