import os
import base64
from pathlib import Path
from flask import Flask
from flask import request
import time

app = Flask(__name__)

@app.route("/")
def server_home():
    return "<p>Server Running!</p>"

@app.route("/upload", methods=["POST"])
def image_upload():
    request_data = request.get_json()
    category = request_data["category"]
    imageData = request_data["imageData"]
    Path(f"./Images/{category}").mkdir(parents=True, exist_ok=True)
    try:
        with open(f"./Images/{category}/{time.time()}.jpeg", "wb") as fh:
            fh.write(base64.decodebytes(imageData.encode()))
    except:
        print("Exception occured")
        return {
            "statusCode": 500
        }

    return {
        "statusCode": 200
    }