#!/usr/bin/env python
# coding: utf-8

from keras.optimizers import SGD
from tensorflow.keras import layers, models

class CNN():
    def __init__(self):
        self.model = models.Sequential()

    def build_cnn(self,act='relu'):
        """
        Builds the actual CNN+LSTM Hybrid Model
        """
        self.model.add(layers.Conv2D(40, (3, 3), activation=act, input_shape=(39, 345, 1), padding="same"))
        self.model.add(layers.MaxPooling2D((3, 3),strides=(2,2),padding='same'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.2))

        self.model.add(layers.Conv2D(80, (3, 3), activation=act, padding="same"))
        self.model.add(layers.MaxPooling2D((3, 3),strides=(2,2),padding='same'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.1))

        self.model.add(layers.Conv2D(160, (3, 3), activation=act, padding="same"))
        self.model.add(layers.MaxPooling2D((3, 3),strides=(2,2),padding='same'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.1))

        self.model.add(layers.Conv2D(160, (3, 3), activation=act, padding="same"))
        self.model.add(layers.MaxPooling2D((1, 44),strides=(1,1),padding="same"))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.1))

        #self.model.add(layers.TimeDistributed(layers.Flatten()))
        #self.model.add(layers.LSTM(128))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64,activation=act))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(32,activation=act))
        self.model.add(layers.Dense(9,activation='softmax'))

        self.model.summary()
        sgd = SGD(learning_rate=0.001)
        self.model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
        
    def train_cnn(self, X_train, y_train, X_val, y_val, epochs=25, batch_size=2):
        """
        Fit the model to the training data.
        """
        print("Training Model")
        self.history = self.model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val),batch_size=batch_size)
    
    def predict(self, X_test):
        """
        Perform predictions.
        """
        prediction = self.model.predict(X_test)
        return prediction
    
