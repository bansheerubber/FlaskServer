import os
import base64
from pathlib import Path
from flask import Flask
from flask import request
import time

import tensorflow as tf
import numpy as np
import cv2
import pickle
from keras import models, layers
from keras.datasets import mnist
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, MaxPool2D
from keras.models import Sequential

app = Flask(__name__)

def get_file_type(image_bytes):
	if image_bytes[0:4] == b"\x89PNG":
		return "png"
	elif image_bytes[0:4] == b"\xFF\xD8\xFF\xE0":
		return "jpg"
	elif image_bytes[0:4] == b"\x52\x49\x46\x46":
		return "webp"
	elif image_bytes[0:3] == b"\x50\x34\x0A":
		return "pbm"
	elif image_bytes[0:4] == b"\x59\xA6\x6A\x95":
		return "ras"
	elif image_bytes[0:4] == b"\x4D\x4D\x00\x2A" or image_bytes[0:4] == b"\x49\x49\x2A\x00":
		return "tiff"
	elif image_bytes[0:4] == b"\x76\x2F\x31\x01":
		return "exr"
	elif image_bytes[0:4] == b"\x23\x3F\x52\x41":
		return "hdr"
	else:
		return "unknown"

def load_model(file_name):
	if not os.path.isfile(file_name):
		(x_train, y_train) , (x_test, y_test) = mnist.load_data()

		x_train = x_train / 255
		x_test = x_test / 255
		x_train = x_train.reshape(-1, 28, 28, 1)
		x_test = x_test.reshape(-1, 28, 28, 1)

		model = models.Sequential()
		model.add(Conv2D(32, kernel_size = (3, 3), activation = "relu", input_shape = (28, 28, 1)))
		model.add(MaxPooling2D((2, 2)))
		model.add(Conv2D(64, kernel_size = (3, 3), activation = "relu"))
		model.add(MaxPooling2D((2, 2)))
		model.add(Conv2D(64, kernel_size = (3, 3), activation = "relu"))
		model.add(MaxPooling2D((2, 2)))
		model.add(Flatten())
		model.add(Dense(64, activation = "relu"))
		model.add(Dense(10, activation = "softmax"))

		model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
		model.fit(x_train, y_train, epochs = 20)
		model.evaluate(x_test, y_test)

		# pickle.dump(model, open(file_name, "wb"))
		model.save(file_name)

	# return pickle.load(open(file_name, "rb"))
		return tf.keras.models.load_model(file_name)

model = load_model("model.h5")
# tf.keras.models.load_model("model.h5")

def predict_digit(model, image_bytes): # takes image file name and classifies it
	img_size = 28

	np_array = np.frombuffer(image_bytes, np.uint8)
	img = cv2.imdecode(np_array, -1)

	grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # greyscale
	grayImage = ~grayImage # invert image like in dataset
	(thresh, grayImage) = cv2.threshold(grayImage, 200, 255, cv2.THRESH_BINARY)
	resizedImage = cv2.resize(grayImage, (img_size, img_size), interpolation = cv2.INTER_AREA) # resize image
	finalImage = tf.keras.utils.normalize(resizedImage, axis = 1)
	finalImage = np.array(finalImage).reshape(-1, img_size, img_size, 1)
	predictions = model.predict(finalImage)
	return np.argmax(predictions)

@app.route("/")
def server_home():
	return "<p>Server Running!</p>"

@app.route("/upload", methods=["POST"])
def image_upload():
	request_data = request.get_json()
	imageData = request_data["imageData"]

	# decode the image
	try:
		decoded = base64.decodebytes(imageData.encode())
		classification = str(predict_digit(model, decoded))
		extension = get_file_type(decoded)
	except:
		print("Could not classify file")
		return {
			"statusCode": 500
		}

	# ensure that we are not overwriting any files
	destination = f"./Images/{classification}"
	Path(destination).mkdir(parents = True, exist_ok = True)
	numFiles = len([name for name in os.listdir(destination) if os.path.isfile(f"{destination}/{name}")])

	potential_name = numFiles + 1
	while os.path.isfile(f"{destination}/{potential_name}.{extension}"):
		potential_name += 1

	# write the file to disk
	try:
		with open(f"{destination}/{potential_name}.{extension}", "wb") as fh:
			fh.write(base64.decodebytes(imageData.encode()))
	except:
		print("Exception occured")
		return {
			"statusCode": 500
		}

	return {
		"statusCode": 200
	}
