RFMODEL_MODEL_NAME = "rf_models/rfmodel.sav"

import os
from angles import get_ang
from pose import get_pose
from PIL import Image
import pickle

model = pickle.load(open(RFMODEL_MODEL_NAME, "rb"))

def get_randomforest_pred(image, model):
    pose = get_pose(image)
    angles = get_ang(pose)
    prediction = model.predict_proba([angles])[0]
    i = prediction.argmax()
    return (int(i), float(prediction[i]))


