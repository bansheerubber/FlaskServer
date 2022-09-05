import json
import os
import io
import PIL.Image as Image
from pathlib import Path
from flask import Flask
from flask import request
from array import array


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
    newFileBytes = bytes(imageData,encoding='utf-8')
    Path(f"./Images/{category}").mkdir(parents=True, exist_ok=True)
    numFiles = len([name for name in os.listdir('.') if os.path.isfile(name)])
    # newFile = open(f"{numFiles+1}.png", "wb")
    # newFile.write(newFileByteArray)

    # bytes = readimage(f"./Images/{category}.png")
    image = Image.open(io.BytesIO(newFileBytes))
    image.save(f"./Images/{category}/{numFiles+1}/.png")
    return {
        "statusCode": 200
    }


def readimage(path):
    count = os.stat(path).st_size / 2
    with open(path, "rb") as f:
        return bytearray(f.read())