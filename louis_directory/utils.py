import keras
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import random

class Utils:

    def __init__(self, image_file_normal=None, image_file_symptoms=None, image=None, limit=None):
        self.image_file_normal = image_file_normal
        self.image_file_symptoms = image_file_symptoms
        self.image = image
        self.limit = limit
        self.file_reader_normal()
        self.file_reader_symptoms()

    def image_reader(self):
        if self.image is not None:
            self.image = cv2.imread(self.image)

    def file_reader_normal(self):
        if self.image_file_normal is not None:
            images_normal = [cv2.imread(file) for file in glob.glob(self.image_file_normal + "*.png")]
            images_normal = random.sample(images_normal, self.limit)
            self.images_normal = images_normal
            return images_normal

    def file_reader_symptoms(self):
        if self.image_file_symptoms is not None:
            if isinstance(self.image_file_symptoms, list):
                symptoms = []
                for directory in self.image_file_symptoms:
                    directory_images = [cv2.imread(file) for file in glob.glob(os.path.join(directory, "*.png"))]
                    symptoms.append(random.sample(directory_images, int(self.limit / len(self.image_file_symptoms))))
                images_symptoms = [item for sublist in symptoms for item in sublist]
            else:
                images_symptoms = [cv2.imread(file) for file in glob.glob(self.image_file_symptoms + "*.png")]
                images_symptoms = random.sample(images_symptoms, self.limit)

            self.images_symptoms = images_symptoms
            return images_symptoms

    def model_picker(self):
        if self.organ == 'lung':
            return lung_model.predict(self.image)
        if self.organ == 'brain':
            return brain_model.predict(self.image)
        if self.organ == 'knee':
            return knee_model.predict(self.image)
        if self.organ == 'shoulder':
            return shoulder_model.predict(self.image)
        if self.organ == 'spine':
            return spine_model.predict(self.image)
