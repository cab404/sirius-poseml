from flask import Flask, request
from PIL import Image
import json, io, os

from ch import pose
# os.chdir(os.path.split(__file__)[0])

app = Flask("mahalovo")

def categorize(image):
    return 1

@app.route('/')
def index():
    return open("static/index.html", "r").read()

@app.route("/api/recognize", methods=["POST"])
def recognize():
    raw_image = request.files["image"].read()
    image = Image.open(io.BytesIO(raw_image))
    return json.dumps({"category": categorize(image)})

app.run(port=8080)
