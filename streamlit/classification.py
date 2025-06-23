import numpy as np
import pandas as pd
import streamlit as st

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

@st.cache_data
def load_data():
   
   """load the iris dataset"""
   iris=load_iris()
   df=pd.DataFrame(iris.data,columns=iris.feature_names)
   df['species']=iris.target
   return df,iris.target_names  
    
df, target_names = load_data()
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(df.drop('species',axis=1),df['species'])
st.title("Iris Species Classification")
sepal_length = st.sidebar.slider("Sepal Length (cm)", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()), float(df['sepal length (cm)'].mean()))
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()), float(df['sepal width (cm)'].mean()))

petal_length = st.sidebar.slider("Petal Length (cm)", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()), float(df['petal length (cm)'].mean()))         
petal_width = st.sidebar.slider("Petal Width (cm)", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()), float(df['petal width (cm)'].mean()))
input_data=[[sepal_length,sepal_width,petal_length,petal_width   ]]


prediction=model.predict(input_data)
predicted_species = target_names[prediction[0]]
st.write(f"Predicted Species: {predicted_species}")

