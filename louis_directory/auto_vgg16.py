import keras
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import BinaryAccuracy, Recall
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split



class AutoVgg16:

    def __init__(self,  normal_files, symptoms_files, limit, epoch, patience):
        self.epoch = epoch
        self.patience = patience

        self.normal_files = normal_files
        self.symptoms_files = symptoms_files
        self.limit = limit

        self.file_reader_normal()
        self.file_reader_symptoms()

        self.labels_normal, self.labels_symptom = self.label_maker()
        self.X_train, self.X_test, self.y_train, self.y_test = self.train_test_images()

        self.model = self.initialize_vgg16_model()
        self.history = self.get_history()

    def file_reader_normal(self):
        images_normal = [cv2.imread(file) for file in glob.glob(self.normal_files+"*.png")]
        images_normal = random.sample(images_normal, self.limit)
        self.images_normal = images_normal

    def file_reader_symptoms(self):
        if self.symptoms_files is not None:
            if isinstance(self.symptoms_files, list):
                symptoms_images = []
                for directory in self.symptoms_files:
                    directory_images = [cv2.imread(file) for file in glob.glob(os.path.join(directory, "*.png"))]
                    symptoms_images.extend(random.sample(directory_images, int(self.limit / len(self.symptoms_files))))
            else:
                symptoms_images = [cv2.imread(file) for file in glob.glob(self.symptoms_files+"*.png")]
                symptoms_images = random.sample(symptoms_images, self.limit)

        self.symptoms_images = symptoms_images



    def label_maker(self):
        labels_normal = [0] * len(self.images_normal)
        labels_symptom = [1] * len(self.symptoms_images)

        return labels_normal, labels_symptom

    def train_test_images(self):
        X = np.concatenate((self.symptoms_images, self.images_normal), axis=0)
        y = np.concatenate((self.labels_symptom, self.labels_normal), axis=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
        return X_train, X_test, y_train, y_test


    def initialize_vgg16_model(self):

        metrics = [BinaryAccuracy(name='binary_accuracy'), Recall(name='recall')]
        optimizer = Adam(learning_rate=0.001)

        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        for layer in base_model.layers:
            layer.trainable = False

        model = Sequential([
        base_model,
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)

        return model


    def get_history(self):

        es = EarlyStopping(patience = self.patience, restore_best_weights=True)

        history = self.model.fit(self.X_train, self.y_train,
            epochs=self.epoch,
            batch_size=64,
            validation_split = 0.2,
            callbacks=[es],
            verbose=1)

        return history

    def plot_loss_accuracy(self, title=None):
        fig, ax = plt.subplots(1,3, figsize=(20,7))

        # --- LOSS ---

        ax[0].plot(self.history.history['loss'])
        ax[0].plot(self.history.history['val_loss'])
        ax[0].set_title('Model loss')
        ax[0].set_ylabel('Loss')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylim((0,3))
        ax[0].legend(['Train', 'Validation'], loc='best')
        ax[0].grid(axis="x",linewidth=0.5)
        ax[0].grid(axis="y",linewidth=0.5)

        # --- ACCURACY

        ax[1].plot(self.history.history['binary_accuracy'])
        ax[1].plot(self.history.history['val_binary_accuracy'])
        ax[1].set_title('Model Accuracy')
        ax[1].set_ylabel('Accuracy')
        ax[1].set_xlabel('Epoch')
        ax[1].legend(['Train', 'Validation'], loc='best')
        ax[1].set_ylim((0,1))
        ax[1].grid(axis="x",linewidth=0.5)
        ax[1].grid(axis="y",linewidth=0.5)

        # --- RECALL

        ax[2].plot(self.history.history['recall'])
        ax[2].plot(self.history.history['val_recall'])
        ax[2].set_title('Model Recall')
        ax[2].set_ylabel('Recall')
        ax[2].set_xlabel('Epoch')
        ax[2].legend(['Train', 'Validation'], loc='best')
        ax[2].set_ylim((0,1))
        ax[2].grid(axis="x",linewidth=0.5)
        ax[2].grid(axis="y",linewidth=0.5)

        if title:
            fig.suptitle(title)

    def evaluate_accuracy(self):

        evaluation = self.model.evaluate(self.X_test, self.y_test)
        accuracy = evaluation[1]
        recall = evaluation[2]

        return f'accuracy:{accuracy}, recall:{recall}'
