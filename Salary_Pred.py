# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import category_encoders as ce

''' Removes Irrelevant columns from Dataset '''
def dropIrrelevantColumns(data) :
    data = data.drop('Wears Glasses', axis = 1)
    data = data.drop('Hair Color', axis = 1)
    data = data.drop('Instance', axis = 1)
    return data

''' Preprocesses Data as commented '''
def preprocessData(data, data_test) :
    #Split Data into Independent and Dependent Variable
    X = pd.DataFrame(data.iloc[:, :-1])
    X_test = pd.DataFrame(data_test.iloc[:, :-1])
    Y = pd.Series(data['Total Yearly Income [EUR]'])

    X['train'] = 1
    X_test['train'] = 0
    cmb = pd.concat([X, X_test])
    del X
    del X_test
    
    cmb['Gender'] = cmb['Gender'].fillna('unknown')
    cmb['Gender'] = cmb['Gender'].replace('0', 'unknown')
    cmb['University Degree'] = cmb['University Degree'].fillna('No')
    cmb['University Degree'] = cmb['University Degree'].replace('0', 'No')
    cmb['Profession'].fillna(cmb['Profession'].mode()[0], inplace=True)
    cmb['Country'].fillna(cmb['Country'].mode()[0], inplace=True)
    cmb['Age'].fillna(cmb['Age'].mean(), inplace=True)
    cmb['Year of Record'].fillna(cmb['Year of Record'].median(), inplace=True)
    cmb['Body Height [cm]'].fillna(cmb['Body Height [cm]'].mean(), inplace=True)
    cmb['Crime Level in the City of Employement'].fillna(cmb['Crime Level in the City of Employement'].mode(), inplace=True)
    cmb['Crime Level in the City of Employement'] = pd.Categorical(cmb['Crime Level in the City of Employement'])
    
    cmb['Housing Situation'] = cmb['Housing Situation'].replace('0', 'nA')
    
    cmb['Work Experience in Current Job [years]'] = cmb['Work Experience in Current Job [years]'].replace('#NUM!', np.nan)
    cmb['Work Experience in Current Job [years]'] = pd.to_numeric(cmb['Work Experience in Current Job [years]'])
    m = cmb['Work Experience in Current Job [years]'].mean()
    print(str(m))
    cmb['Work Experience in Current Job [years]'].fillna(m, inplace=True)
    cmb['Satisfation with employer'].fillna(cmb['Satisfation with employer'].mode()[0], inplace=True)
    cmb['Yearly Income in addition to Salary (e.g. Rental Income)'] = cmb.apply(
            lambda row: float(row['Yearly Income in addition to Salary (e.g. Rental Income)'].split()[0]), axis=1
            )
    '''cmb['Size of City'] = cmb.apply(
        lambda row: cmb['Size of City'].where(cmb['Country']==row['Country'] & cmb['Year of Record']==row['Year of Record']).mean() if np.isnan(row['Size of City']) else row['Size of City'],
        axis=1
    )'''
    cmb['Size of City'].fillna(cmb['Size of City'].mean, inplace=True)
    
    #cmb[cmb['Size of City'].isnull()][nullSize][cmb['Country']]
    #cmb['Size of City'].fillna(cmb['Size of City'].mean(), inplace=True)
    #cmb['Wears Glasses'] = cmb['Wears Glasses'].fillna(cmb['Wears Glasses'].mode()[0])
    #cmb['Hair Color'] = cmb['Hair Color'].fillna(cmb['Hair Color'].mode()[0])
    '''
    le_G = LabelEncoder()
    cmb['Gender'] = le_G.fit_transform(cmb['Gender'])
    
    le_D = LabelEncoder()
    cmb['University Degree'] = le_D.fit_transform(cmb['University Degree'])
    
    le_P = LabelEncoder()
    cmb['Profession'] = le_P.fit_transform(cmb['Profession'])
    
    le_C = LabelEncoder()
    cmb['Country'] = le_C.fit_transform(cmb['Country'])
    
    ohe = OneHotEncoder(categories='auto', sparse=False)
    ohe = ohe.fit(cmb.iloc[:,[1,3,5,6]])
    '''
    
    #cmb.dropna()
    te = ce.TargetEncoder()
    
    X = cmb[cmb['train'] == 1]
    X_test = cmb[cmb['train'] == 0]
    Z = pd.Series(X['Yearly Income in addition to Salary (e.g. Rental Income)'])
    Z_test = pd.Series(X_test['Yearly Income in addition to Salary (e.g. Rental Income)'])
    X = X.drop('Yearly Income in addition to Salary (e.g. Rental Income)', axis=1)
    X_test = X_test.drop('Yearly Income in addition to Salary (e.g. Rental Income)', axis=1)
    X = X.drop('train', axis=1)
    X_test = X_test.drop('train', axis=1)
    X = te.fit_transform(X, Y, verbose = 1000)
    X_test = te.transform(X_test)
    
    
    '''
    X_t = ohe.transform(X.iloc[:,[1,3,5,6]])
    X_test_t = ohe.transform(X_test.iloc[:,[1,3,5,6]])
    X = X.merge(pd.DataFrame(X_t), how='outer', left_index=True, right_index=True)
    X_test = X_test.merge(pd.DataFrame(X_test_t), how='outer', left_index=True, right_index=True)
    X.drop(['train'], axis=1, inplace=True)
    X_test.drop(['train'], axis=1, inplace=True)
    
    sc = MinMaxScaler()
    X = pd.DataFrame(sc.fit_transform(X))
    X_test = pd.DataFrame(sc.transform(X_test))
    '''
    Y = Y.subtract(Z)
    return (X,Y, X_test, Z_test)

data = pd.read_csv('data.csv')
data_test = pd.read_csv('data_test.csv')
data = dropIrrelevantColumns(data)
data_test = dropIrrelevantColumns(data_test)
X , Y , X_test, Z_test = preprocessData(data, data_test)
'''
sc_y = MinMaxScaler()
sc_y = sc_y.fit(Y)
Y = sc_y.transform(Y)
'''

'''X = X.drop(X.columns[2], axis = 1)
X = X.drop(X.columns[3], axis = 1)
X_test = X_test.drop(X_test.columns[2], axis = 1)
X_test = X_test.drop(X_test.columns[3], axis = 1)'''

#Iteration 1, CatBoostEncoder
#regressor = DecisionTreeRegressor()
#regressor = SVR()
#regressor = LinearRegression(n_jobs=-1)
regressor = RandomForestRegressor(n_estimators=50, verbose=1000, n_jobs=-1)
#from sklearn.linear_model import BayesianRidge
#regressor = BayesianRidge(verbose=True)
#from sklearn.linear_model import SGDRegressor
#regressor = SGDRegressor(verbose=1)

#from sklearn.linear_model import Ridge
#regressor = Ridge()
regressor.fit(X, Y)
Y_pred = regressor.predict(X_test)
#Y_pred = sc_y.inverse_transform(Y_pred.reshape(-1,1))
Y_pred = np.array(Y_pred)
Z_test = np.array(Z_test)
Y_pred = np.add(Y_pred, Z_test)
with open("ypredTE.csv", "w") as file:
    for i in np.array(Y_pred) :
        file.write(str(i) + "\n")
        
