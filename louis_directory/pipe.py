import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

class Pipe:

    def __init__(self, file_path):
        self.file_path = file_path

    def image_reader(self):
        if self.file_path is not None:
            self.image_array= cv2.imread(self.file_path)
            self.image_array = tf.expand_dims(self.image_array, axis=0)

    # come back to check which class classifies what
    def get_organ(self):
        self.image_reader()
        organ_classifier =  load_model('../models/organ_detection_model_final.h5')
        self.organ = None

        if self.image_array is not None:
            which_organ = organ_classifier.predict(self.image_array)
            which_organ = np.argmax(which_organ)

            print(which_organ)
            if which_organ == 0:
                self.organ = 'knee'
            elif which_organ == 1:
                self.organ = 'brain'
            elif which_organ == 2:
                self.organ = 'shoulder'
            elif which_organ == 3:
                self.organ = 'spine'
            elif which_organ == 4:
                self.organ = 'lung'

        return self.organ

    def disese_or_not(self):
        self.organ = self.get_organ()

        if self.organ == 'lung':
            disease_model =  load_model('../models/lung_bin_model.h5') # will be load_model instead

        elif self.organ == 'brain':
            disease_model= load_model('../models/brain_bin_model.h5')

        elif self.organ == 'knee':
            disease_model=  load_model('../models/knee_bin_model.h5')

        elif self.organ == 'shoulder':
            disease_model = load_model('../models/shoulder_bin_model.h5')

        elif self.organ == 'spine':
            disease_model  = load_model('../models/base_spine_bin_model.h5')

        self.disease = disease_model.predict(self.image_array)[0][0]
        print(self.disease)

    def which_disease(self):
        self.disese_or_not()
        if self.disease == 1:
            if self.organ == 'lung':
                lung_cl_model = load_model('../models/lung_class_model.h5')
                return 'lung disease prediction: ' + str(lung_cl_model.predict(self.image_array))
            elif self.organ == 'brain':
                brain_cl_model = load_model('../models/brain_class_model.h5')
                return 'brain disease prediction: ' + str(brain_cl_model.predict(self.image_array))
            elif self.organ == 'knee':
                knee_cl_model = load_model('../models/knee_class_model.h5')
                return 'knee disease prediction: ' + str(knee_cl_model.predict(self.image_array))
            elif self.organ == 'shoulder':
                shoulder_cl_model = load_model('../models/shoulder_class_model.h5')
                return 'shoulder disease prediction: ' + str(shoulder_cl_model.predict(self.image_array))
            elif self.organ == 'spine':
                spine_cl_model = load_model('../models/spine_class_model.h5')
                return 'spine disease prediction: ' + str(spine_cl_model.predict(self.image_array))
        else:
            return 'is healthy'
