#!/usr/bin/env python
# coding: utf-8

# LOGREG_MODELS_DIR = "lr_models"
LOGREG_MODEL_NAME = "lr_models/vlad.sav"
import os
from angles import get_ang
from pose import get_pose
from PIL import Image
import pickle

model = pickle.load(open(LOGREG_MODEL_NAME, "rb"))


def get_logreg_pred(image, model):
    pose = get_pose(image)
    angles = get_ang(pose)
    prediction = model.predict_proba([angles])[0]
    i = prediction.argmax()
    return (int(i), float(prediction[i]))
