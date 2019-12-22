#!/usr/bin/env python
# coding: utf-8

TREEDEC_MODEL_NAME = "td_models/tree_dec.sav"
import os
from angles import get_ang, clip_pose
from pose import get_pose
from PIL import Image
import pickle

treedec_model = pickle.load(open(TREEDEC_MODEL_NAME, "rb"))


def get_treedec_pred(image, model):
    pose = get_pose(image)
    pose = clip_pose(pose)
    angles = get_ang(pose)
    prediction = model.predict_proba([angles])[0]
    i = prediction.argmax()
    return (int(i), float(prediction[i]))
