import os
import math
import numpy as np
import pandas as pd
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Activation,Dense, Dropout

TRAIN_FILE_PATH='./data/training_data(1000).xlsx'
TEST_FILE_PATH='./data/testing_data.xlsx'

def PreprocessData(raw_df):
    df = raw_df.drop(['name'], axis=1)
    age_mean = df['age'].mean()
    df['age'] = df['age'].fillna(age_mean)
    fare_mean = df['fare'].mean()
    df['fare'] = df['fare'].fillna(fare_mean)
    df['sex'] = df['sex'].map({'female': 0, 'male': 1}).astype(int)
    boat_mean = df['boat']
    boat_mean_size = boat_mean.size
    count_of_boat = 0
#
    while(count_of_boat < boat_mean_size):
        boat_mean_with_count = boat_mean[count_of_boat]
        if type(boat_mean_with_count) == str:
            boat_mean[count_of_boat] = 2
        else:
            if math.isnan(float(boat_mean_with_count)):
                boat_mean[count_of_boat] = 0
            else:
                boat_mean[count_of_boat] = 1
        count_of_boat = count_of_boat + 1

    x_OneHot_df = pd.get_dummies(data=df, columns = ["embarked"])
    ndarray = x_OneHot_df.values
    label = ndarray[:,0] 
    Features = ndarray[:,1:] 
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaledFeatures = minmax_scale.fit_transform(Features)
    return scaledFeatures, label
#
train_df = pd.read_excel(TRAIN_FILE_PATH)
test_df = pd.read_excel(TEST_FILE_PATH)
#
cols = ['survived', 'pclass', 'sex','name', 'age', 'sibsp', 'parch', 'fare', 'embarked','boat']
train_df = train_df[cols]
test_df = test_df[cols]
#
train_result, train_label = PreprocessData(train_df)
test_feature, test_label = PreprocessData(test_df)
#
print("len(test_feature) " , type(test_feature))
print("len(test_feature) " , len(test_feature))
print("len(test_label) " , type(test_label))
print("len(test_label) " , len(test_label))
#
model = Sequential()
model.add(Dense(units=80, input_dim=10, kernel_initializer='uniform'))
model.add(Activation('relu'))
model.add(Dense(units=60, kernel_initializer='uniform'))
model.add(Activation('relu'))
model.add(Dense(units=1, kernel_initializer='uniform'))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(x = train_result, y = train_label, epochs = 200, validation_split = 0.05, batch_size = 5, verbose = 1)
#
scores = model.evaluate(x = train_result, y = train_label, batch_size=10)
print(scores)
res = model.predict(test_feature, batch_size=5, verbose=0)
ids = 0
with open('result.csv', 'a') as out:
    out.write("id,survived" + '\n')
    for value in res:
        if value[0] > 0.5:
            out.write(str(ids) + "," + str(1) + '\n')
        else:
            out.write(str(ids) + "," + str(0) + '\n')
        ids += 1

