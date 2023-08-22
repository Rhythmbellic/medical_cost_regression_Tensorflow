#importing libraries
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#Read the insurance dataset
insurance=pd.read_csv('https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv')
insurance.head()
#scaling our data and one hot encoding
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
ct= make_column_transformer((MinMaxScaler(),['age','bmi','children']),(OneHotEncoder(handle_unknown='ignore'),['sex','smoker','region']))
#create x and y labels
X=insurance.drop('charges',axis=1)
y=insurance['charges']
#creating training and test dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
#fitting column transformer
ct.fit(X_train)
X_train_normal=ct.transform(X_train)
X_test_normal=ct.transform(X_test)
#building a neural network
tf.random.set_seed(42)
model=tf.keras.Sequential([
    tf.keras.layers.Dense(100,activation='relu'),
    tf.keras.layers.Dense(50,activation='relu'),
    tf.keras.layers.Dense(10,activation='relu'),
    tf.keras.layers.Dense(1)
])
#compiling the model
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['mae'])
#fit the model
history=model.fit(X_train_normal,y_train,epochs=200)
#plotting loss curve
pd.DataFrame(history.history).plot()
#predicting the result
y_pred=model.predict(X_test_normal)
y_test,y_pred
