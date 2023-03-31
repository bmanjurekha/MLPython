import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder 

dataset = pd.read_csv("insurance.csv")
print(dataset.head())

lencoder = LabelEncoder()

dataset[['sex','smoker','region']] = dataset[['sex','smoker','region']].apply(lambda col: lencoder.fit_transform(col))

X = dataset[["age","sex","bmi","children","smoker","region"]]

y = dataset['charges']

#Spliting the data for training(80%) & testing (20%)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

#Applying Linear Regression ML Algorithm

regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

pickle.dump(regression_model, open('model.pkl','wb'))
