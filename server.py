import os
import base64
from pathlib import Path
from flask import Flask
from flask import request

app = Flask(__name__)

@app.route("/")
def server_home():
    return "<p>Server Running!</p>"

@app.route("/upload", methods=["POST"])
def image_upload():
    request_data = request.get_json()
    category = request_data["category"]
    imageData = request_data["imageData"]
    print('str',imageData)
    Path(f"./Images/{category}").mkdir(parents=True, exist_ok=True)
    numFiles = len([name for name in os.listdir('.') if os.path.isfile(name)])
    try:
        with open(f"./Images/{category}/{numFiles+1}.jpeg", "wb") as fh:
            fh.write(base64.decodebytes(imageData.encode()))
    except:
        print("Exception occured")
        return {
            "statusCode": 500
        }

    return {
        "statusCode": 200
    }