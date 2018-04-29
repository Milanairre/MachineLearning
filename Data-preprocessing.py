# -*- coding: utf-8 -*-
"""
Data pre-processing, imputing missing values (mean imputation) , train-test split and
label encoding of the categorical variables
"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""import datasets
Anytime you have to import the datasets, you need to setup the working directory
and that must be the folder that contains your datasets
How to CD to a diff drive using jupyter?
go to windows command line > cd /d d:\Learn

Using Spyder IDE
use the top Menu
"""

"""setup working directory : put the file .py to the folder where raw data is saves
and run and that directory becomes the working directory"""

dataset = pd.read_csv('Data.csv')

"""salary appears as G, scientific notation so we change the data type to float by applying
 0g after %. i.e %.0f"""
 
"""now we need to put the independant variables to matrix i.e first 3 col as x
and then we will create a dependant variables vector which is the purchased column"""

"""# [left of the comma is row and right is column] """
X = dataset.iloc[:, :-1].values   
y = dataset.iloc[:, 3] 



"""handling missing data Age and Salary
We impute the data with the mean.  Press ctrl i to take the hint"""
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN' , strategy ='mean' , axis = 0)

"""fit the imputer object to the X"""
imputer.fit(X[:, 1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#Categorical Variables are country and Purchased so we need to encode them to numbers
#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
labelencoder_X.fit_transform(X [:,0]) 
X[:,0] = labelencoder_X.fit_transform(X [:,0])
#inspect the class by pressing ctrl + i
onehotencoder = OneHotEncoder(categorical_features= [0])
X=onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y=labelencoder_y.fit_transform(y)

"""split the dataset into train test split"""
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y , test_size = 0.2 , random_state = 0)


#feature Scaling

#feature Scaling
#What is feature scaling?
'''Age and Salary contains numbers but not in the same scale as Age is going from 27 to 50
and salary is going from 42,000 to 90,000. 
ML models are based Euclidean Distance and since there is a big gap between Age and Salary, Euclidean distance is
dominated by salary'''

'''There are two types of features scaling: Standardiation and Normalisation

standardisation= value-mean(all value)/standard deviation(all value)
normalisation= value-min(all values)/max(all value)-min(all value)


With these techniques, we are putting our variables on the same scale so that no variables dominates other
'''


#practical Feature Scaling on Age and Salary
#let's start by scaling independant variables

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test= sc_X.transform(X_test)











