#!/usr/bin/env python
# coding: utf-8

LOGREG_MODELS_DIR = "lg_models"

import os
from angles import get_ang, clip_pose
from sklearn.linear_model import LogisticRegression
import pickle

def load_classifier(fname: str) -> LogisticRegression:
    m = pickle.load(open(fname, 'rb'))
    return m

logreg_models = list([
    load_classifier(os.path.join(LOGREG_MODELS_DIR, model))
    for model
    in os.listdir(LOGREG_MODELS_DIR)
])

import numpy as np
from pose import get_pose
from PIL import Image

def get_logreg_pred(image, classifier):
    pose = get_pose(image)
	pose = clip_pose(pose)
    angles = get_ang(pose)
    prediction = classifier.predict_proba((np.array(angles)).reshape(1, -1))
    i = prediction[0].argmax()
    return (int(i), float(prediction[0][i]))
