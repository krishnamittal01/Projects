import streamlit as st
import pandas as pd
st.title("Streamlit Information  Input")
name=st.text_input("Enter your Name:")

if name:
    st.write(f"Hello {name} welcome to Streamlit!")

age=st.slider("Select your age:",0,100,25)
st.write(f"You are {age} years old.")

options=st.selectbox("Select your favorite color:",["Red", "Green", "Blue"])
st.write(f"You selected {options} as your favorite color.")

    
# Create a simple data frame
data=({
    "first column": [1, 2, 3, 4],
    "second column": [10, 20, 30, 40]
})

d=pd.DataFrame(data)
st.write("Here is our first dataframe:",d)



##MAking file upload button


uploaded_file=st.file_uploader("Choose a csv file",type="csv")

if uploaded_file is not None:
    df=pd.read_csv(uploaded_file,index_col=[0])
    st.write("Here is the uploaded dataframe:")
    st.write(df)