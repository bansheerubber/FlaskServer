import os
import base64
from pathlib import Path
from flask import Flask,current_app
from flask import request
import time

import json
import tensorflow as tf
import numpy as np
import cv2
import pickle
from keras import models, layers
from keras.datasets import mnist
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, MaxPool2D, BatchNormalization
from keras.models import Sequential
totalPred =  [0] * 10
totalClassification = [""] * 10
counter = 0
cmodel = ""
topleft = ""
topright = ""
bottomleft = ""
bottomright = ""
classification = -1
counter = 0
app = Flask(__name__)

@app.before_first_request
def load_model():
	global cmodel
	global topleft
	global topright
	global bottomleft
	global bottomright
	# file_name = "model.h5"
	file_name = "trained_model_topleft.h5"
	if not os.path.isfile(file_name):
		(x_train, y_train) , (x_test, y_test) = mnist.load_data()

		x_train = x_train / 255
		x_test = x_test / 255
		x_train = x_train.reshape(-1, 28, 28, 1)
		x_test = x_test.reshape(-1, 28, 28, 1)

		trained_model_topleft = models.Sequential()

		trained_model_topleft.add(Conv2D(32,(3,3),strides = (1,1), input_shape = (28, 28, 1)))
		trained_model_topleft.add(BatchNormalization(axis=3))
		trained_model_topleft.add(Activation('relu'))
		trained_model_topleft.add(Conv2D(32,(3,3),strides = (1,1)))
		trained_model_topleft.add(BatchNormalization(axis=3))
		trained_model_topleft.add(Activation('relu'))
		trained_model_topleft.add(MaxPooling2D((2,2),strides = (2,2)))
		trained_model_topleft.add(Conv2D(64,(3,3),strides = (1,1)))
		trained_model_topleft.add(BatchNormalization(axis=3))
		trained_model_topleft.add(Activation('relu'))
		trained_model_topleft.add(Conv2D(64,(3,3),strides = (1,1)))
		trained_model_topleft.add(BatchNormalization(axis=3))
		trained_model_topleft.add(Activation('relu'))
		trained_model_topleft.add(MaxPooling2D((2,2),strides = (2,2)))
		trained_model_topleft.add(Dropout(0.2))
		trained_model_topleft.add(Flatten())
		trained_model_topleft.add(Dense(256,activation = 'relu'))
		trained_model_topleft.add(Dropout(0.4))
		trained_model_topleft.add(Dense(10,activation = 'softmax'))
		trained_model_topleft.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
		trained_model_topleft.fit(x_train, y_train, epochs = 20)
		trained_model_topleft.evaluate(x_test, y_test)

		# pickle.dump(model, open(file_name, "wb"))
		trained_model_topleft.save(file_name)

		trained_model_topright = models.Sequential()

		trained_model_topright.add(Conv2D(32,(3,3),strides = (1,1), input_shape = (28, 28, 1)))
		trained_model_topright.add(BatchNormalization(axis=3))
		trained_model_topright.add(Activation('relu'))
		trained_model_topright.add(Conv2D(32,(3,3),strides = (1,1)))
		trained_model_topright.add(BatchNormalization(axis=3))
		trained_model_topright.add(Activation('relu'))
		trained_model_topright.add(MaxPooling2D((2,2),strides = (2,2)))
		trained_model_topright.add(Conv2D(64,(3,3),strides = (1,1)))
		trained_model_topright.add(BatchNormalization(axis=3))
		trained_model_topright.add(Activation('relu'))
		trained_model_topright.add(Conv2D(64,(3,3),strides = (1,1)))
		trained_model_topright.add(BatchNormalization(axis=3))
		trained_model_topright.add(Activation('relu'))
		trained_model_topright.add(MaxPooling2D((2,2),strides = (2,2)))
		trained_model_topright.add(Dropout(0.2))
		trained_model_topright.add(Flatten())
		trained_model_topright.add(Dense(256,activation = 'relu'))
		trained_model_topright.add(Dropout(0.4))
		trained_model_topright.add(Dense(10,activation = 'softmax'))
		trained_model_topright.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
		trained_model_topright.fit(x_train, y_train, epochs = 20)
		trained_model_topright.evaluate(x_test, y_test)

		# pickle.dump(model, open(file_name, "wb"))
		trained_model_topright.save(file_name)

		trained_model_bottomleft = models.Sequential()

		trained_model_bottomleft.add(Conv2D(32,(3,3),strides = (1,1), input_shape = (28, 28, 1)))
		trained_model_bottomleft.add(BatchNormalization(axis=3))
		trained_model_bottomleft.add(Activation('relu'))
		trained_model_bottomleft.add(Conv2D(32,(3,3),strides = (1,1)))
		trained_model_bottomleft.add(BatchNormalization(axis=3))
		trained_model_bottomleft.add(Activation('relu'))
		trained_model_bottomleft.add(MaxPooling2D((2,2),strides = (2,2)))
		trained_model_bottomleft.add(Conv2D(64,(3,3),strides = (1,1)))
		trained_model_bottomleft.add(BatchNormalization(axis=3))
		trained_model_bottomleft.add(Activation('relu'))
		trained_model_bottomleft.add(Conv2D(64,(3,3),strides = (1,1)))
		trained_model_bottomleft.add(BatchNormalization(axis=3))
		trained_model_bottomleft.add(Activation('relu'))
		trained_model_bottomleft.add(MaxPooling2D((2,2),strides = (2,2)))
		trained_model_bottomleft.add(Dropout(0.2))
		trained_model_bottomleft.add(Flatten())
		trained_model_bottomleft.add(Dense(256,activation = 'relu'))
		trained_model_bottomleft.add(Dropout(0.4))
		trained_model_bottomleft.add(Dense(10,activation = 'softmax'))
		trained_model_bottomleft.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
		trained_model_bottomleft.fit(x_train, y_train, epochs = 20)
		trained_model_bottomleft.evaluate(x_test, y_test)

		# pickle.dump(model, open(file_name, "wb"))
		trained_model_bottomleft.save(file_name)

		trained_model_bottomright = models.Sequential()

		trained_model_bottomright.add(Conv2D(32,(3,3),strides = (1,1), input_shape = (28, 28, 1)))
		trained_model_bottomright.add(BatchNormalization(axis=3))
		trained_model_bottomright.add(Activation('relu'))
		trained_model_bottomright.add(Conv2D(32,(3,3),strides = (1,1)))
		trained_model_bottomright.add(BatchNormalization(axis=3))
		trained_model_bottomright.add(Activation('relu'))
		trained_model_bottomright.add(MaxPooling2D((2,2),strides = (2,2)))
		trained_model_bottomright.add(Conv2D(64,(3,3),strides = (1,1)))
		trained_model_bottomright.add(BatchNormalization(axis=3))
		trained_model_bottomright.add(Activation('relu'))
		trained_model_bottomright.add(Conv2D(64,(3,3),strides = (1,1)))
		trained_model_bottomright.add(BatchNormalization(axis=3))
		trained_model_bottomright.add(Activation('relu'))
		trained_model_bottomright.add(MaxPooling2D((2,2),strides = (2,2)))
		trained_model_bottomright.add(Dropout(0.2))
		trained_model_bottomright.add(Flatten())
		trained_model_bottomright.add(Dense(256,activation = 'relu'))
		trained_model_bottomright.add(Dropout(0.4))
		trained_model_bottomright.add(Dense(10,activation = 'softmax'))
		trained_model_bottomright.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
		trained_model_bottomright.fit(x_train, y_train, epochs = 20)
		trained_model_bottomright.evaluate(x_test, y_test)

		# pickle.dump(model, open(file_name, "wb"))
		trained_model_bottomright.save(file_name)

	# return pickle.load(open(file_name, "rb"))
		# return tf.keras.models.load_model(file_name)
	# cmodel = tf.keras.models.load_model(file_name)
	topleft = tf.keras.models.load_model(file_name)
	topright = tf.keras.models.load_model("trained_model_topright.h5")
	bottomleft = tf.keras.models.load_model("trained_model_bottomleft.h5")
	bottomright = tf.keras.models.load_model("trained_model_bottomright.h5")


	# model = load_model("model.h5")

# handlers.py
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


# model = load_model("model.h5")

# tf.keras.models.load_model("model.h5")

def predict_digit(model, image_bytes): # takes image file name and classifies it
	global counter,totalPred
	img_size = 14

	np_array = np.frombuffer(image_bytes, np.uint8)
	img = cv2.imdecode(np_array, -1)
	# print("img.shape", img.shape)
	img = cv2.resize(img,(img_size,img_size))
	grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # greyscale
	# cv2.imshow("Gray",grayImage)
	# cv2.waitKey(0) 
  
	# #closing all open windows 
	# cv2.destroyAllWindows() 
	
	# print("grayImage.shape", grayImage.shape)
	grayImage = ~grayImage # invert image like in dataset
	(thresh, grayImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
	resizedImage = cv2.resize(grayImage, (img_size, img_size), interpolation = cv2.INTER_AREA) # resize image
	finalImage = tf.keras.utils.normalize(resizedImage, axis = 1)
	finalImage = np.array(finalImage).reshape(-1, img_size, img_size, 1)
	predictions = model.predict(finalImage)
	return predictions

def getModel(position):
	if position == 0:
		return topleft
	elif position == 1:
		return topright
	elif position == 2:
		return bottomleft
	else:
		return bottomright

@app.route("/")
def server_home():
	return "<p>Server Running!</p>"

@app.route("/infer", methods=["POST"])
def image_upload():
	global cmodel
	global classification
	global totalPred
	global counter
	global totalClassification
	request_data = request.get_json()
	imageData = request_data["imageData"]
	position = request_data["position"]

	# decode the image
	try:
		decoded = base64.decodebytes(imageData.encode())
		# print("decoded: ",decoded)
		
		# print("cmodel",cmodel)
		predictions = predict_digit(getModel(position=position), decoded)
		classification = str(np.argmax(predictions))
		# print("position: ",position,"list: ",predictions.tolist())
		
		
		extension = get_file_type(decoded)
	except Exception as e: 
		print (str(e))
		print("Could not classify file")
		return {
			"statusCode": 500
		}

	# # ensure that we are not overwriting any files
	# destination = f"./Images/{position}"
	# Path(destination).mkdir(parents = True, exist_ok = True)
	# numFiles = len([name for name in os.listdir(destination) if os.path.isfile(f"{destination}/{name}")])

	# potential_name = numFiles + 1
	# while os.path.isfile(f"{destination}/{potential_name}.{extension}"):
	# 	potential_name += 1

	# write the file to disk
	# try:
	# 	with open(f"{destination}/{potential_name}.{extension}", "wb") as fh:
	# 		fh.write(base64.decodebytes(imageData.encode()))
	# except:
	# 	print("Exception occured")
	# 	return {
	# 		"statusCode": 500
	# 	}

	return {
		"statusCode": 200,
		"predictions": predictions.tolist(),
		"classification": classification,
		"position": position
	}
