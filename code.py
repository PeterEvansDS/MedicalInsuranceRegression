#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 11:32:26 2021

@author: peterevans
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv(r'/Users/peterevans/UniCloud/Scripting/Assignment2/insurance.csv')

#%% a) explore basic information of dataset and check if there are any issues

print(data.info())

#check each column for odd inputs
print('\nSex unique values: ', data.sex.unique())
print('Smoker unique values: ', data.smoker.unique())
print('Region unique values: ', data.region.unique())
print("Age MAX: ", data.age.max(), " Age MIN: ", data.age.min())
print("BMI MAX: ", data.bmi.max(), " BMI MIN: ", data.bmi.min())
print("Children MAX: ", data.children.max(), " Children MIN: ", data.children.min())
print("Charges MAX: ", data.charges.max(), " Charges MIN: ", data.charges.min())

#find any duplicate data entries
duplicates = data.duplicated(keep=False)
print('\n', data[duplicates])
data.drop([0, 581], inplace=True)

#%% b) generate more data

#generate age category
age_bins = [18,24,55,float('inf')]
age_labels = ['Young', 'Adult', 'Elder']
data['age_cat'] = pd.cut(data.age, bins = age_bins, 
                         labels = age_labels, include_lowest=True)

#generate weight category
bmi_bins = [float('-inf'),18.5,25,30,float('inf')]
weight_labels = ['Underweight', 'Normal weight', 'Overweight', 'Obese']
data['weight'] = pd.cut(data.bmi, bins = bmi_bins, labels = weight_labels)

#%% c) compare the number of smokers and non-smokers between genders

sex_smokers = pd.DataFrame(data.groupby(['smoker', 'sex']).size().reset_index(
                                                                name='number'))
print(sex_smokers)

sns.set_theme()
sns.barplot(data = sex_smokers, x = 'sex', y = 'number', hue = 'smoker')
plt.tight_layout()
plt.show()

#%% d i) convert categorical data to numerical and save it

data.sex = data.sex.replace({'female':1, 'male':0})
data.smoker = data.smoker.replace({'yes':1, 'no':0})
dummies = pd.get_dummies(data.region, drop_first = True)
data[['northwest', 'southeast', 'southwest']] = dummies
data.drop(['region'], axis=1, inplace = True)

data.to_csv(r'/Users/peterevans/UniCloud/Scripting/Assignment2/data_cleaned.csv')

#%% d) ii) analyse the relationships between charges feature and other features

#correlation matrix
print('CORRELATION: \n', data.corr().to_string())
print('\nCORRELATION WITH CHARGES: \n', data.corr()['charges'].sort_values(
                                                            ascending=False))
#find p-values 
import statsmodels.formula.api as smf
lin_mod = smf.ols(formula='charges ~ age + sex + bmi + children + smoker +\
              northwest + southeast + southwest', data=data).fit()
print('\nP-VALUES: \n', lin_mod.pvalues)

#forward search using adj-R^2
X = data[['age', 'sex', 'bmi', 'children', 'smoker', 
          'northwest', 'southeast', 'southwest']]
y = data.charges

test_cols = X.columns.tolist()
chosen_cols= []
adj_R2_final = []

for i in range(len(X.columns)):
    adj_R2_array = []
    formula1 = 'charges ~'
    for j in chosen_cols:
        formula1 += (j + '+')
    for k in test_cols:
        formula2 = formula1 + k
        lin_mod = smf.ols(formula=formula2, data=data).fit()
        adj_R2_array.append(lin_mod.rsquared_adj)
    adj_R2_final.append(max(adj_R2_array))
    chosen_cols.append(test_cols[adj_R2_array.index(max(adj_R2_array))])
    test_cols.remove(test_cols[adj_R2_array.index(max(adj_R2_array))])

results = pd.DataFrame({'Added feature': chosen_cols, 'AdjR2 Value': adj_R2_final}, 
                       index = [1,2,3,4,5,6,7,8])
results.index.name = '# of Features'
print('\nFORWARD SEARCH RESULTS: \n', results)

#plot data to see relationship between features and response
X = data[['age', 'sex', 'bmi', 'children', 'smoker', 
          'northwest', 'southeast', 'southwest']]
y = data.charges
fig, axs = plt.subplots(2,4)
for i in range(2):
    for j in range(4):
        axs[i,j].scatter(X.iloc[:, (4*i+j)], y)
        axs[i, j].set(xlabel='{}'.format(X.columns[4*i+j]))

#plt smoking and age
plt.figure()
sns.scatterplot(data = data, x = 'age', y = 'charges', hue = 'smoker')
plt.tight_layout()
plt.show()

#%% e) build an estimator with your selected features to estimate charges
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

X = data[['age', 'bmi', 'children', 'smoker', 'southeast', 'southwest']]
y = data['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=0)

columns = ['Estimator', 'Training RMSE', 'Training R2', 'Test RMSE', 'Test R2']
performance = pd.DataFrame(columns=columns)

#function to record scores
def return_metrics(name, model, X1, X2):
    y_pred_train = model.predict(X1)
    y_pred_test = model.predict(X2)
    r2_train = r2_score(y_train, y_pred_train)
    rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
    result = pd.DataFrame([[name, rmse_train, r2_train, rmse_test, r2_test]], 
                          columns = columns)
    return result


#simple linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
performance  = performance.append(return_metrics('Ordinary Least Squares', 
                                                 lin_reg, X_train, X_test))

#linear fit with polynomial terms
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)
lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_train_poly, y_train)
performance  = performance.append(return_metrics('Least Squares with Polynomial',
                                                 lin_reg_poly, X_train_poly, X_test_poly))

#ridge regression
from sklearn.linear_model import RidgeCV
ridge = RidgeCV(alphas = np.arange(0.1,1,0.01))
ridge.fit(X_train, y_train)
y_ridge = ridge.predict(X_test)
print('RIDGE LAMBDA = ', ridge.alpha_)
performance  = performance.append(return_metrics('Ridge', ridge, X_train, X_test))

#lasso regression
from sklearn.linear_model import LassoCV
lasso = LassoCV(eps = 1e-5)
lasso.fit(X_train, y_train)
y_lasso = lasso.predict(X_test)
print('LASSO LAMBDA = ', lasso.alpha_)
performance  = performance.append(return_metrics('Lasso', lasso, X_train, X_test))

#ridge poly
ridge_poly = RidgeCV(alphas=np.arange(0.1,1,0.01))
ridge_poly.fit(X_train_poly, y_train)
y_ridge_poly = ridge_poly.predict(X_test_poly)
print('RIDGE POLY LAMBDA = ', ridge_poly.alpha_)

performance  = performance.append(return_metrics('Ridge with Polynomial', ridge_poly,
                                                 X_train_poly, X_test_poly))
#lasso poly
lasso_poly = LassoCV(random_state=0, normalize=True)
lasso_poly.fit(X_train_poly, y_train)
y_lasso_poly = lasso_poly.predict(X_test_poly)
print('LASSO POLY LAMBDA = ', lasso_poly.alpha_, '\n')

performance  = performance.append(return_metrics('Lasso with Polynomial', lasso_poly,
                                                 X_train_poly, X_test_poly))

print(performance.to_string())

#save all models
import joblib
joblib.dump(lin_reg, r'/Users/peterevans/UniCloud/Scripting/Assignment2/lin_reg.pkl')
joblib.dump(lin_reg_poly, r'/Users/peterevans/UniCloud/Scripting/Assignment2/lin_reg_poly.pkl')
joblib.dump(ridge, r'/Users/peterevans/UniCloud/Scripting/Assignment2/ridge.pkl')
joblib.dump(ridge_poly, r'/Users/peterevans/UniCloud/Scripting/Assignment2/ridge_poly.pkl')
joblib.dump(lasso, r'/Users/peterevans/UniCloud/Scripting/Assignment2/lasso.pkl')
joblib.dump(lasso_poly, r'/Users/peterevans/UniCloud/Scripting/Assignment2/lasso_poly.pkl')

#%% e) ii) visualise estimators with testing data and predicted values

fig1 = plt.figure(1)
plt.scatter(X_test.age, y_test, label = 'True')
plt.scatter(X_test.age, lin_reg.predict(X_test), label = 'Predicted')
plt.ylabel('charges')
plt.xlabel('age')
plt.title('Ordinary Least Squares')
plt.legend()
plt.show()

fig1 = plt.figure(2)
plt.scatter(X_test.age, y_test, label = 'True')
plt.scatter(X_test.age, lin_reg_poly.predict(X_test_poly), label = 'Predicted')
plt.ylabel('charges')
plt.xlabel('age')
plt.title('OLS with Polynomial terms')
plt.legend()
plt.show()

fig3 = plt.figure(3)
plt.scatter(y_test, abs(y_test - lin_reg.predict(X_test)), label = 'OLS')
plt.scatter(y_test, abs(y_test - lin_reg_poly.predict(X_test_poly)), label = 'OLS Poly')
plt.ylabel('Residual')
plt.xlabel('True Y Value')
plt.legend()
plt.show()

fig4 = plt.figure(4)
plt.scatter(X_test.age, y_test, label = 'True')
plt.scatter(X_test.age, ridge_poly.predict(X_test_poly), label = 'Predicted')
plt.ylabel('charges')
plt.xlabel('age')
plt.title('Ridge Regression with Polynomial')
plt.legend()
plt.show()



