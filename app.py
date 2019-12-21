from flask import Flask, request
from PIL import Image
import json, io, os
from pose import get_pose

# os.chdir(os.path.split(__file__)[0])

app = Flask("mahalovo")

def categorize(image):
    return (1, 0.33)

@app.route('/')
def index():
    return open("static/index.html", "r").read()

@app.route("/api/recognize", methods=["POST"])
def recognize():
    raw_image = request.files["image"].read()
    image = Image.open(io.BytesIO(raw_image))
    pred, acc = categorize(image)
    return json.dumps({"pred": pred, "acc": acc})

app.run(host='0.0.0.0', port=8080)
