import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score #to predict the accuracy of model 
import pickle 


#DATA COLLECTION AND PROCESSING

#LOADING DATASET TO PANDAS FRAMEWORK
loan_dataset=pd.read_csv(r"Loan_prediction_dataset.csv")

#droping missing values
loan_dataset=loan_dataset.dropna()

#label enconding
loan_dataset=loan_dataset.replace({"Loan_Status":{'no':0,'yes':1}})

#replacing the value of 3+ dependents to 4
loan_dataset=loan_dataset.replace(to_replace='3+',value=4)

#convert categorical columns to numerical values
loan_dataset.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}}, inplace=True)

#seperating the data and label
X=loan_dataset.drop(columns=['Loan_ID','Loan_Status'],axis=1)
Y=loan_dataset['Loan_Status']

#train test split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.01,stratify=Y,random_state=1)

#training model
classifier=svm.SVC(kernel='linear')
classifier.fit(X_train,Y_train)

#accuracy
X_train_prediction=classifier.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print('accuracy:',training_data_accuracy)

#Prediction
input_data = (1,1,1,1,0,5955,5625,315,360,1,2)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = classifier.predict(input_data_reshaped) 


if(prediction=='N'):
        print("Loan is not Approved")
else:
        print("Loan is Approved")
 

 
# saving the trained model


filename='trained_model.sav'
pickle.dump(classifier,open(filename,'wb')) # write a binary file 

#loading the saved model



loaded_model=pickle.load(open('trained_model.sav','rb'))




