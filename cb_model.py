#!/usr/bin/env python
# coding: utf-8

CATBOOST_MODELS_DIR = "cb_models"

import os
from angles import get_ang
from catboost import *

def load_classifier(fname: str) -> CatBoostClassifier:
    m = CatBoostClassifier()
    m.load_model(fname)
    return m

catboost_models = list([
    load_classifier(os.path.join(CATBOOST_MODELS_DIR, model))
    for model
    in os.listdir(CATBOOST_MODELS_DIR)
])

from pose import get_pose
from PIL import Image

def get_catboost_pred(image, classifier):
    pose = get_pose(image)
	pose = clip_pose(pose)
    angles = get_ang(pose)
    prediction = classifier.predict_proba(angles)
    i = prediction.argmax()
    return (int(i), float(prediction[i]))
