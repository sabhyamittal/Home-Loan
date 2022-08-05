import numpy as np
import pickle
import streamlit as st


loaded_model=pickle.load(open("C:\\Users\\sabhya mittal\\OneDrive\\Desktop\\Home Prediction\\trained_model.sav",'rb'))


st.title("Home Loan Prediction Using Machine Learning")
st.header("Fill your details")
def prediction(input_data):
    
   input_data_as_numpy_array = np.asarray(input_data)
   input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
   prediction = loaded_model.predict(input_data_reshaped)
   

   if(prediction=='N'):
        return 'Loan is not Approved'
   else:
        return  'Loan is  Approved'
    
def main():
    
    st.title("Home Loan Prediction Using Machine Learning")
    st.header("Fill your details")
    
a= st.selectbox('What is your Gender?',('Male', 'Female'))
if a=='Male':
  aa=1
elif a=='Female':
     aa=0

b = st.selectbox('Are you Married?',('No', 'Yes'))
if b=='Yes':
    bb=1
elif b=='No':
    bb=0

c = st.selectbox(' Are you Graduate?',('No', 'Yes'))
if c=='Yes':
    cc=1
elif c=='No':
    cc=0

d = st.selectbox('What is your town identified as?',('Rural', 'Semi-Urban', 'Urban'))
if d=='Rural':
    dd=0
elif d=='Semi-Urban':
    dd=1
elif d=='Urban':
    dd=2

j = st.selectbox('Does the applicant have a healthy credit history?',('No', 'Yes'))
if j=='Yes':
    jj=1
elif j=='No':
    jj=0

k = st.selectbox('Are you Self-Employed?',('No', 'Yes'))
if k=='Yes':
    kk=1
elif k=='No':
    kk=0

e = st.number_input('Number of Dependents')
f = st.number_input('Applicants Income')
g = st.number_input('Co-Applicants Income')
h = st.number_input('Loan Amount applying for')
i = st.number_input('Loan Amount Term')
 
#code for prediction
result=''

#prediction

if st.button('predict'):
    result=prediction([aa, bb, e, cc, kk, f, g, h, i, jj, dd])
    
    st.success(result) #st.success displays a success message.

    if __name__ == '_main_':
        
        main()
        
    
     
    
        

