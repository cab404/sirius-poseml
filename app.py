from flask import Flask, request
from PIL import Image
import json, io, os
import logging

log = logging.getLogger("mahalovo")
log.level = logging.INFO
log.info("[...] Starting PoseML")

log.info("[...] Pose recognizer...")
from pose import get_pose
# log.info("[...] Catboost model...")
# from cb_model import get_catboost_pred, catboost_models
log.info("[...] Random Forest Classifier model...")
from rf_model import get_randomforest_pred, rf_model
log.info("[...] Logistic Regression model...")
from lr_model import get_logreg_pred, logreg_model
log.info("[...] Tree Decision model...")
from td_model import get_treedec_pred, treedec_model
log.info("[+++] Complete! Starting server.")

app = Flask("mahalovo")

def treedec_categorize(image):
    pred, acc = get_treedec_pred(image, treedec_model)
    log.info(f"[TreeDec] Predicted {pred} with P={acc}")
    return (pred, acc)

def logreg_categorize(image):
    pred, acc = get_logreg_pred(image, logreg_model)
    log.info(f"[LogReg] Predicted {pred} with P={acc} !")
    return (pred, acc)

def catboost_categorize(image):
    pred, acc = get_catboost_pred(image, catboost_models[1])
    log.info(f"[Catboost] Predicted {pred} with P={acc} !")
    return (pred, acc)

def dummy_categorize(image):
    _ = input().split()
    pred, acc = int(_[0]), float(_[1])
    log.info(f'[Dummy] Predicted {pred} with P={acc} !')
    return (pred, acc)

def randomforest_categorize(image):
    pred, acc = get_randomforest_pred(image, rf_model)
    log.info(f"[Random Forest] Predicted {pred} with P={acc} !")
    return (pred, acc)

categorize = randomforest_categorize

@app.route('/')
def index():
    return open("static/index.html", "r").read()

@app.route('/static/chart.png')
def chart():
    return open('static/chart.png', 'rb').read()

@app.route('/static/favicon.ico')
def favicon():
    return open('static/favicon.ico', 'rb').read()

@app.route("/api/recognize", methods=["POST"])
def recognize():
    raw_image = request.files["image"].read()
    image = Image.open(io.BytesIO(raw_image))
    pred, acc = categorize(image)
    return json.dumps({"pred": pred, "acc": acc})

app.run(host='0.0.0.0', port=8080)

