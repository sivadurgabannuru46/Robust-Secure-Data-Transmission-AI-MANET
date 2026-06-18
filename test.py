import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ABC import ABC
from SwarmPackagePy import testFunctions as tf
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout, Flatten
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
import os
import pickle

dataset = pd.read_csv("AODVDataset/AODV.csv")
dataset.fillna(0, inplace = True)

label_encoder = []
columns = dataset.columns
types = dataset.dtypes.values
for i in range(len(types)):
    name = types[i]
    if name == 'object': #finding column with object type
        le = LabelEncoder()
        dataset[columns[i]] = pd.Series(le.fit_transform(dataset[columns[i]].astype(str)))#encode all str columns to numeric 
        label_encoder.append(le)

dataset = dataset.values
X = dataset[:,0:dataset.shape[1]-1]
Y = dataset[:,dataset.shape[1]-1]

scaler = MinMaxScaler(feature_range = (0, 1)) #use to normalize training features
X = scaler.fit_transform(X)
print(X.shape)
alh = ABC(X, tf.easom_function, -10, 10, 2, 20)
Gbest = np.asarray(alh.get_Gbest())
print(Gbest)
print(Gbest.shape)

in_mask = [True if i > 0 else False for i in Gbest]
in_mask = np.asarray(in_mask)
X_selected_features = X[:,in_mask==1] 
print(X_selected_features.shape)


svm_cls = SVC(probability=True)
svm_cls.fit(X_selected_features, Y)
predict = svm_cls.predict_proba(X_selected_features)
print(predict)
print(predict.shape)
Y = to_categorical(Y)
X_selected_features = np.reshape(X_selected_features, (X_selected_features.shape[0], X_selected_features.shape[1], 1))
X_train, X_test, y_train, y_test = train_test_split(X_selected_features, Y, test_size=0.2)
X_train, X_test1, y_train, y_test1 = train_test_split(X_selected_features, Y, test_size=0.1)
'''
ann_model = Sequential()
ann_model.add(Dense(512, input_dim=X_train.shape[1], activation='relu', kernel_initializer = "uniform"))
ann_model.add(Dense(512, activation='relu', kernel_initializer = "uniform"))
ann_model.add(Dense(y_train.shape[1], activation='softmax',kernel_initializer = "uniform"))
'''
ann_model = Sequential()
ann_model.add(Flatten(input_shape=[X_train.shape[1],X_train.shape[2]]))
ann_model.add(Dense(300, activation="relu"))
ann_model.add(Dense(100, activation="relu"))
ann_model.add(Dense(y_train.shape[1], activation="softmax"))
ann_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
if os.path.exists("model/1model_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/model_weights.hdf5', verbose = 1, save_best_only = True)
    hist = ann_model.fit(X_selected_features, Y, batch_size = 32, epochs = 350, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    ann_model.load_weights("model/model_weights.hdf5")


