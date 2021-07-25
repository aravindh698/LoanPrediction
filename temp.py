import pandas as pd

import pickle

df2=pd.read_csv(r'C:\Users\ARAVIND\Downloads\test_lAUu6dG.csv',index_col='Loan_ID')
df1=pd.read_csv(r'C:\Users\ARAVIND\Downloads\train_ctrUa4K.csv',index_col='Loan_ID')
df1['Gender'].fillna('Male',inplace=True)
df1['Married'].fillna('Yes',inplace=True)
df1['Self_Employed'].fillna('No',inplace=True)
df1['LoanAmount'].fillna(120.0,inplace=True)
df1['Loan_Amount_Term'].fillna(360.0,inplace=True)
df1['Credit_History'].fillna(1.0,inplace=True)
df1['Gender'].fillna('Male',inplace=True)
df1['Married'].fillna('Yes',inplace=True)
df1['Self_Employed'].fillna('No',inplace=True)
df1['LoanAmount'].fillna(120.0,inplace=True)
df1['Loan_Amount_Term'].fillna(360.0,inplace=True)
df1['Credit_History'].fillna(1.0,inplace=True)
df2['Gender'].fillna('Male',inplace=True)
df2['Married'].fillna('Yes',inplace=True)
df2['Self_Employed'].fillna('No',inplace=True)
df2['LoanAmount'].fillna(120.0,inplace=True)
df2['Loan_Amount_Term'].fillna(360.0,inplace=True)
df2['Credit_History'].fillna(1.0,inplace=True)
df1['Income']=df1['ApplicantIncome']+df1['CoapplicantIncome']
df2['Income']=df2['ApplicantIncome']+df2['CoapplicantIncome']
df1['Income']=(df1['Income']-df1['Income'].min())/(df1['Income'].max()-df1['Income'].min())
df2['Income']=(df2['Income']-df2['Income'].min())/(df2['Income'].max()-df2['Income'].min())
df1['LoanAmount']=(df1['LoanAmount']-df1['LoanAmount'].min())/(df1['LoanAmount'].max()-df1['LoanAmount'].min())
df2['LoanAmount']=(df2['LoanAmount']-df2['LoanAmount'].min())/(df2['LoanAmount'].max()-df2['LoanAmount'].min())
df1['Loan_Amount_Trem']=(df1['Loan_Amount_Term']-df1['Loan_Amount_Term'].min())/(df1['Loan_Amount_Term'].max()-df1['Loan_Amount_Term'].min())
df2['Loan_Amount_Trem']=(df2['Loan_Amount_Term']-df2['Loan_Amount_Term'].min())/(df2['Loan_Amount_Term'].max()-df2['Loan_Amount_Term'].min())
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1['Gender']=le.fit_transform(df1['Gender'])
df1['Married']=le.fit_transform(df1['Married'])
df1['Education']=le.fit_transform(df1['Education'])
df1['Self_Employed']=le.fit_transform(df1['Self_Employed'])
df1['Property_Area']=le.fit_transform(df1['Property_Area'])
df1['Loan_Status']=le.fit_transform(df1['Loan_Status'])
df2['Gender']=le.fit_transform(df2['Gender'])
df2['Married']=le.fit_transform(df2['Married'])
df2['Education']=le.fit_transform(df2['Education'])
df2['Self_Employed']=le.fit_transform(df2['Self_Employed'])
df2['Property_Area']=le.fit_transform(df2['Property_Area'])
y=df1.Loan_Status
features=['Married','Property_Area','Credit_History']
X=df1[features].copy()
X_test=df2[features].copy()
from sklearn.model_selection import train_test_split
X_train,X_valid,y_train,y_valid=train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=0)
from sklearn.svm import SVC
sv=SVC(C=1.0,kernel='rbf')
sv.fit(X_train,y_train)
pickle.dump(sv, open('model.pkl','wb'))
f= open('model.pkl','rb')
model=pickle.load(f)
print(model.predict([[1, 1, 1]]))
