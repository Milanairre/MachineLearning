# -*- coding: utf-8 -*-
"""
Simple Linear Regression

@author: Milan
"""

'''LR is a line that best fits our data using OLS methods and how every unit change affets the dependant variable
Constant is when the x value is zero, what is a Y value.
Eg: Salary = 30k when the person has zero experience
Slope is what is how unit increase affects the Y or salary variable 

How simple linear regression finds that best fit line?

Yi = actual value
Yi hat= is the modelled value, line from regression

What regression does is it sums the squared (yi-yihat) and records it and then looks for minimum distance/line
OR minimum sum of squares 
'''

#simple Linear Regression

# Step 1: Data Preprocessing

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the datasets

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values   
y = dataset.iloc[:, 1] 

#splitting the data into training ans test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y , test_size = 1/3 , random_state = 0)

#feature Scaling
# for simple regression, we dont need to scale the features as the libraries will do the 
#job for us
'''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test= sc_X.transform(X_test)
sc_y = StandardScaler()
y_train= sc_y.fit.transform(y_train)
'''


#fitting the Simple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#use fit method of the class
regressor.fit(X_train, y_train)


#predicting the test results
y_pred = regressor.predict(X_test)

#visualizing the traiing set results
plt.scatter (X_train, y_train , color= 'red')
plt.plot(X_train, regressor.predict(X_train) , color ='blue')
plt.title( 'Salary vs Experience (Training set)')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()



#visualizing the test set results
plt.scatter (X_test, y_test , color= 'red')
plt.plot(X_train, regressor.predict(X_train) , color ='blue')
plt.title( 'Salary vs Experience (Test set)')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()




















