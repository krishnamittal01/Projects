import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


##Title of he application

st.title("Hello Streamlit")

##Display a simple text
st.write('This is a  simple streamlit application')

## create a simple data frame


df=pd.DataFrame({
    "first column": [1, 2, 3, 4],
    "second column": [10, 20, 30, 40]
})

##Display the dataframe

st.write("Here is our first dataframe:")
st.write(df)


##create a line chart
chart_data = pd.DataFrame(
    np.random.randn(20,3),columns=['a', 'b', 'c']
)
st.line_chart(chart_data)
