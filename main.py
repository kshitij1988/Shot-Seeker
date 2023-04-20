#!/usr/bin/env python
# coding: utf-8

import os
import sys
import copy
import random
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.math import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing as p
from utils import *
from model import CNN
import threading
from keras.utils import np_utils

def get_data(data_path, hop_length=128, nfft=1024, n_classes=9):
    loader = Loader("Reading Data...", "Reading Data... Done", 0.05).start()
    list_folder = os.listdir(data_path)
    raw_data_x = []
    raw_data_y = []
    for gun_type in list_folder:
        wav_file_list = os.listdir(os.path.join(data_path,gun_type))
        for wav_file in wav_file_list:
            filename = os.path.join(data_path, gun_type, wav_file)
            wav_data, sr = librosa.load(filename,sr=22050)
            raw_data_x.append(wav_data)
            raw_data_y.append(gun_type)
    loader.stop()

    loader = Loader("Augmenting Data...", "Augmenting Data... Done", 0.05).start()
    #Data Augmentation
    aug_data_x = augment_data(raw_data_x)
    loader.stop()

    loader = Loader("Extracting Audio Features...", "Extracting Audio Features... Done", 0.05).start()
    #Audio Feature generation
    x = []
    y = []
    for j in range(0,4,1):
        for i in range(len(raw_data_x)):
            data = aug_data_x[j][i]
            if len(data) == 44100:
                MFCCs = librosa.feature.mfcc(y=data, sr=sr, n_fft=nfft, hop_length=hop_length,n_mfcc=13)
                delta_mfcc = librosa.feature.delta(MFCCs)
                delta_mfcc2 = librosa.feature.delta(MFCCs,order=2)
                MFCC_total = np.concatenate((MFCCs,delta_mfcc,delta_mfcc2))
                x.append(MFCC_total)
                y.append(raw_data_y[i])

    #Encode labels and get test, train, val splits
    y_copy = copy.copy(y)
    le = p.LabelEncoder()
    le.fit(y_copy)
    y_copy_new = le.transform(y_copy)
    y_copy_new =  np_utils.to_categorical(y_copy_new, n_classes)
    x_copy = np.array(x)
    loader.stop()

    print("Generating Train Test splits...")
    X_train, X_test, y_train, y_test = train_test_split(x_copy, y_copy_new, test_size=0.25, random_state=123, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=123)

    return X_train, X_test, X_val, y_train, y_test, y_val


def augment_data(data):
    praw,graw,nraw = [], [], []
    for i in range(len(data)):
        praw.append(invert_polarity(data[i]))
        graw.append(random_gain(data[i], -1, 1))
        nraw.append(noise(data[i],0.1))

    aug_data = [data,praw,graw,nraw]
    return aug_data

if __name__ == '__main__':
    data_path = r"{}".format(sys.argv[1])
    
    X_train, X_test, X_val, y_train, y_test, y_val = get_data(data_path)
    
    print("Building the CNN+LSTM Model...")
    cnn = CNN()
    cnn.build_cnn()

    print("Training the Model...")
    cnn.train_cnn(X_train,y_train,X_val,y_val)

    print("Generating Predictions")
    y_hat=cnn.predict(X_test)

    print("Plotting Confusion Matrix")
    plt.imshow(confusion_matrix(np.argmax(y_test,axis=1), np.argmax(y_hat,axis=1)))
    plt.colorbar()
    plt.show()
