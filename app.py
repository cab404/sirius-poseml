from flask import Flask, request
from PIL import Image

app = Flask("mahalovo")

@app.route("/api/recognize", methods=["POST"])
def recognize():
    imgdata = request.get_data()
    image = Image.frombytes("RGB", (320, 240), imgdata)
    return "test"

app.run(port=8080)