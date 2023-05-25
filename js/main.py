import numpy as np
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

loan_train = pd.read_csv('dataset/train_csv.csv')
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
best_params = rf_random.best_params_ 
print(f'Best parameters: {best_params}')
y_pred1 = rf_random.predict(X_test)
print(f'Accuracy: {np. sum(y_pred1==y_test)/len(y_test)}')

def feature_imp(df, model):
    feat = pd.DataFrame(columns=['feature', 'importance']) 
    feat["feature"] = df.columns
    feat["importance"] = model.best_estimator_.feature_importances_ 
    print("here")
    return feat.sort_values(by="importance", ascending=False)
# sns.countplot(loan_train['Dependents'])
# pyplt.show()