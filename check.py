#!/usr/bin/env python
# coding: utf-8

CATBOOST_MODELS_DIR = "cb_models"
LOGREG_MODELS_DIR = "lg_models"

import os
from catboost import *

os.chdir(os.path.join(os.path.split(__file__)[:-1]))

def load_classifier(fname: str) -> CatBoostClassifier:
    m = CatBoostClassifier()
    m.load_model(fname)
    return m

catboost_models = list([
    load_classifier(os.path.join(CATBOOST_MODELS_DIR, model)) 
    for model 
    in os.listdir(CATBOOST_MODELS_DIR)
])

# TF_CUDNN_USE_AUTOTUNE=0

import sys

os.chdir("pose-tensorflow")
from util.config import load_config
from nnet import predict
from util import visualize
from dataset.pose_dataset import data_to_input
os.chdir("..")

import numpy as np

def get_ang(pose):
    features = []
    for i in range(len(pose)):
        for j in range(len(pose)):
            if j != i:
                for k in range(len(pose)):
                    if k > i and k != j:
                        a = pose[i, :-1]
                        b = pose[j, :-1]
                        c = pose[k, :-1]
                        ba = a - b
                        bc = c - b
                        ang = np.arccos(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)))
                        features.append(ang)
    return features

from PIL import Image

def resize_image(img: Image):
    basewidth = 300

    wpercent = basewidth / img.size[0]
    hsize = int(img.size[1] * wpercent)
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    
    return img
    
def init():
    d = {}
    d['cfg'] = load_config("pose-tensorflow/demo/pose_cfg.yaml")
    d['sess'], d['inputs'], d['outputs'] = predict.setup_pose_prediction(d['cfg'])
    return d

def get_pose(image, d):
    image = resize_image(image)
    
    image_batch = data_to_input(image)
    
    outputs_np = d['sess'].run(d['outputs'], feed_dict={d['inputs']: image_batch})
    scmap, locref, _ = predict.extract_cnn_output(outputs_np, d['cfg'])
    
    pose = predict.argmax_pose_predict(scmap, locref, d['cfg'].stride)
    return pose

def get_catboost_pred(image):
    pose = get_pose(image, cfg)
    angles = get_ang(pose)
    return int(m.predict(angles).item())

cfg = init()
