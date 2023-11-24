
from models import
import cv2

class Pipe:

    def __init__(self, image):
        self.image = image


    def image_reader(self):
        if self.image is not None:
            self.image = cv2.imread(self.image)

    # come back to check which class classifes what
    def get_organ(self):
        organ_classifier = load_the_classfier_model()
        self.organ = None

        if self.image is not None:
            which_organ = organ_classifier.predict(self.image)

            if which_organ == 0:
                self.organ = 'lung'
            elif which_organ == 1:
                self.organ = 'brain'
            elif which_organ == 2:
                self.organ = 'knee'
            elif which_organ == 3:
                self.organ = 'shoulder'
            elif which_organ == 4:
                self.organ = 'spine'

        return self.organ

    def disese_or_not(self):

        if self.organ == 'lung':
            lung_model = LungModelClass() # will be load_model instead
            disease_model = lung_model()

        elif self.organ == 'brain':
            brain_model = BrainModelClass()
            disease_model = brain_model()


        elif self.organ == 'knee':
            knee_model = KneeModelClass()
            disease_model = knee_model()

        elif self.organ == 'shoulder':
            shoulder_model = ShoulderModelClass()
            disease_model = shoulder_model()

        elif self.organ == 'spine':
            spine_model = SpineModelClass()
            disease_model = spine_model()

        self.disease = disease_model.predict(self.image)
        return self.disease


    def which_disease(self):
        if self.disease == 1:
            if self.organ == 'lung':
                return 'lung disease prediction: ' + str(lung_disease_classifier.predict(self.image))
            elif self.organ == 'brain':
                return 'brain disease prediction: ' + str(brain_disease_classifier.predict(self.image))
            elif self.organ == 'knee':
                return 'knee disease prediction: ' + str(knee_disease_classifier.predict(self.image))
            elif self.organ == 'shoulder':
                return 'shoulder disease prediction: ' + str(shoulder_disease_classifier.predict(self.image))
            elif self.organ == 'spine':
                return 'spine disease prediction: ' + str(spine_disease_classifier.predict(self.image))
        else:
            return 'is healthy'
