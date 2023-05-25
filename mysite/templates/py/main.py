import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as pyplt
import seaborn as sns

import os
file_path = os.path.join(os.path.dirname(__file__), 'train_csv.csv')
print(file_path)
loan_train= pd.read_csv(file_path)
print(loan_train.shape)
loan_train.head()
print("Variables")
print(loan_train)

total_null = loan_train.isnull().sum().sort_values(ascending=False)
total_null.head(10)

loan_train['Gender'] = loan_train['Gender'].fillna( loan_train['Gender'].dropna( ).mode().values[0] ) 
loan_train[ 'Married'] = loan_train['Married'].fillna( loan_train['Married'].dropna ( ).mode().values[0]) 
loan_train['Dependents'] = loan_train [ 'Dependents'].fillna(
loan_train['Dependents'].dropna( ).mode().values[0] )
loan_train['Self_Employed'] = loan_train['Self_Employed'].fillna(loan_train['Self_Employed'].dropna( ).mode().values[0])
loan_train['LoanAmount'] = loan_train['LoanAmount'].fillna(loan_train['LoanAmount'].dropna( ).mean())
loan_train['Loan_Amount_Term'] = loan_train['Loan_Amount_Term'].fillna( loan_train['Loan_Amount_Term'].dropna( ).mode().values[0])
loan_train['Credit_History'] = loan_train['Credit_History'].fillna( loan_train['Credit_History'].dropna( ).mode().values[0])

loan_train.info()

print(set(loan_train['Gender'].values.tolist()))
print(set(loan_train['Dependents'].values.tolist()))
print(set(loan_train['Married'].values.tolist()))
print(set(loan_train['Education'].values.tolist()))
print(set(loan_train['Self_Employed'].values.tolist()))
print(set(loan_train['Loan_Status'].values.tolist())) 
print(set(loan_train['Property_Area'].values.tolist()))

loan_train['Loan_Status'] = loan_train ['Loan_Status'].map({'N': 0, 'Y':1}).astype(int)
loan_train = pd.get_dummies (loan_train, columns = ['Gender', 'Dependents', 'Married', 'Education', 'Self_Employed', 'Property_Area'])
standardScaler = StandardScaler()
columns_to_scale = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'] 
loan_train [columns_to_scale] = standardScaler.fit_transform(loan_train [columns_to_scale])


y = loan_train [ 'Loan_Status']
X = loan_train.drop(['Loan_Status', 'Loan_ID'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state =42)



rf_param_grid = {
'n_estimators': range(1,1000, 100),
}
rf = RandomForestClassifier()

rf_random = RandomizedSearchCV(param_distributions=rf_param_grid,estimator=rf, scoring ="accuracy", verbose = 0, n_iter = 10, cv = 4)
rf_random.fit(X_train,y_train)
print(list(loan_train.columns))


import joblib
# m = os.path.join(os.path.dirname(__file__))
# print(m)
# joblib.dump(rf_random,m)
# print("here")
# # Load the saved model back into memory
m = os.path.join(os.path.dirname(__file__),"my_model.pkl")
loaded_model = joblib.load(m)
data = {'Loan_ID': [122],
        'Gender': ['Female'],
        'Married': ['Yes'],
        'Dependents': [2],
        'Education': ['Graduate'],
        'Self_Employed': ['Yes'],
        'ApplicantIncome': [340000],
        'CoapplicantIncome': [0],
        'LoanAmount': [503000],
        'Loan_Amount_Term': [1],
        'Credit_History': [550],
        'Property_Area': ['Urban'],
        'Loan_Status': ['N']}

prediction_data = pd.DataFrame(data)
prediction_data['Gender'] = prediction_data['Gender'].fillna(prediction_data['Gender'].dropna().mode().values[0])
prediction_data['Married'] = prediction_data['Married'].fillna(prediction_data['Married'].dropna().mode().values[0])
prediction_data['Dependents'] = prediction_data['Dependents'].fillna(prediction_data['Dependents'].dropna().mode().values[0])
prediction_data['Self_Employed'] = prediction_data['Self_Employed'].fillna(prediction_data['Self_Employed'].dropna().mode().values[0])
prediction_data['LoanAmount'] = prediction_data['LoanAmount'].fillna(prediction_data['LoanAmount'].dropna().mean())
prediction_data['Loan_Amount_Term'] = prediction_data['Loan_Amount_Term'].fillna(prediction_data['Loan_Amount_Term'].dropna().mode().values[0])
prediction_data['Credit_History'] = prediction_data['Credit_History'].fillna(prediction_data['Credit_History'].dropna().mode().values[0])


# Preprocess the data
prediction_data['Education'] = prediction_data['Education'].map({'Graduate': 1, 'Not Graduate': 0})
prediction_data['Self_Employed'] = prediction_data['Self_Employed'].map({'Yes': 1, 'No': 0})
prediction_data['Property_Area'] = prediction_data['Property_Area'].map({'Rural': 0, 'Semiurban': 1, 'Urban': 2})

# One-hot encode the categorical features
categorical_cols = ['Gender', 'Married', 'Dependents']
prediction_data = pd.get_dummies(prediction_data, columns=categorical_cols)

# Scale the numerical features
columns_to_scale = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
scaler = StandardScaler()
prediction_data[columns_to_scale] = scaler.fit_transform(prediction_data[columns_to_scale])
print(prediction_data)
# input_data = np.array([[50000, 700, 0.4, 100000]])

# # Make predictions using the model
from django.shortcuts import render

predictions = loaded_model.predict(prediction_data)
print(predictions)
# # Output the predictions
def loan_eligibility(request):
    if predictions[0] == 1:
        print("Loan approved!")
        eligibility="Loan approved"
        ans='y'
        return render(request, 'index.html', {'result': eligibility,'ans':ans})
    else:
        print("Loan not approved.")
        eligibility="Loan not approved."
        ans='n'
        return render(request, 'index.html', {'result': eligibility,'ans':ans})
