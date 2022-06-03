
#%%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

df = pd.read_csv("https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bikes.csv")
df.shape
#%%

# main question: how many bikes are predicted to 
# be used depending on the day of the week

# df.isna().sum()
df.head(8)


# Transform the dteday column 
df['dteday'] = df['dteday'].str.split(pat="/")
split_df = pd.DataFrame(df['dteday'].tolist(), columns=['month', 'day' , 'year'])
df = pd.concat([df,split_df] , axis=1 , join='inner')
df = df.drop(columns=['dteday'])
df['month'] = df['month'].astype(str).astype(int)
df['day'] = df['day'].astype(str).astype(int)
df['year'] = df['year'].astype(str).astype(int)



df["y"] = df["casual"] + df["registered"]
# df["day"] = df["dteday"].str.substring(0,df["dteday"].str.index("/")
# df["month"]
# df["year"] = df["dteday"]

df.info()
X =  df.drop(["y"], axis = 1)
y = df['y'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 17)

minmax_scaler = preprocessing.MinMaxScaler()
X_train = minmax_scaler.fit_transform(X_train) 
X_test = minmax_scaler.transform(X_test)

#%%

model = tf.keras.models.Sequential()

model.add(Dense(16, input_dim=14, activation='relu'))


# Note: We want the input dimension to match the 
# number of features at our input layer

model.add(Dense(8, activation = 'relu'))

model.add(Dense(1, activation='linear'))

model.compile(loss='MSE', optimizer= 'Adam', metrics=['mean_squared_error'])

#Train the model
history = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 300, verbose = 0)
_, train_mse = model.evaluate(X_train, y_train, verbose = 1)

# %%
y