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
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, MaxPool2D, BatchNormalization
from keras.models import Sequential

from tensorflow.keras.layers.experimental import preprocessing

from PIL import Image

app = Flask(__name__)

def load_image(file_name):
	image = Image.open(file_name)
	pixels = image.load()
	width, height = image.size

	rows = []
	for x in range(0, width):
		column = []
		for y in range(0, height):
			pixel = pixels[y, x][0] / 255
			column.append(pixel)
		rows.append(column)

	return np.asarray(rows)

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

def print_image(image, divisor = 1):
	# print top border
	for x in range(0, len(image) + 2):
		print("-", end = "")
	print("")

	for x in range(0, len(image)):
		print("|", end = "") # left border
		for y in range(0, len(image[x])):
			if image[x][y] > 128 / divisor:
				print("#", end = "")
			elif image[x][y] > 0:
				print(".", end = "")
			else:
				print(" ", end = "")

		print("|") # right border

	# print bottom border
	for x in range(0, len(image) + 2):
		print("-", end = "")
	print("")

# quadrant table:
# 0: top left
# 1: bottom left
# 2: top right
# 3: bottom right
def crop_image(image, quadrant):
	image_half = int(len(image) / 2)

	start_x = quadrant % 2 * image_half
	end_x = start_x + image_half

	start_y = int(quadrant / 2) * image_half
	end_y = start_y + image_half

	rows = []
	for i in range(start_x, end_x):
		rows.append(image[i][start_y:end_y])

	return np.asarray(rows)

def crop_set(x_set, y_set, quadrant_range, debug_level):
	if debug_level >= 1:
		print("Cropping set...")

	cropped_x_set = []
	new_y_set = []
	for i in range(0, len(x_set)):
		image = x_set[i]
		classification = y_set[i]
		if debug_level >= 2:
			print_image(image)

		for i in quadrant_range:
			cropped = crop_image(image, i)
			cropped_x_set.append(cropped)
			new_y_set.append(classification)

			if debug_level >= 3:
				print_image(cropped)

	if debug_level >= 1:
		print("Done cropping set")

	x_set = np.asarray(cropped_x_set)
	x_set[x_set < 160] = 0
	x_set[x_set >= 160] = 255

	return (x_set, np.asarray(new_y_set))

def test_consensus_accuracy(model, x_test, y_test):
	print("Testing model using group consensus:")

	predictions = model.predict(x_test)
	correct = 0
	incorrect = 0
	for i in range(0, len(predictions), 4):
		prediction = np.argmax(
			np.bincount(np.argmax(
					predictions[i:i + 4],
					axis = 1
			))
		)

		if prediction == y_test[i]:
			correct += 1
		else:
			incorrect += 1

	accuracy = correct / (correct + incorrect)
	print(f"Consensus accuracy: {accuracy}")

def load_separate_models(file_name, debug_level = 0):
	results = []
	predictions_results = []

	(_, _), (_, correct_results) = mnist.load_data()

	for i in range(0, 4):
		separate_file_name = f"/home/me/School/Fall 2022/CSE535/FlaskServer/{file_name}.{i}"
		if not os.path.exists(separate_file_name):
			(x_train, y_train), (x_test, y_test) = mnist.load_data()

			(x_train, y_train) = crop_set(x_train, y_train, range(i, i + 1), debug_level)
			(x_test, y_test) = crop_set(x_test, y_test, range(i, i + 1), debug_level)

			image_size = 14

			x_train = x_train / 255
			x_test = x_test / 255
			x_train = x_train.reshape(-1, image_size, image_size, 1)
			x_test = x_test.reshape(-1, image_size, image_size, 1)

			model = models.Sequential()

			model.add(Conv2D(64, kernel_size = (3, 3), activation = "relu", input_shape = (image_size, image_size, 1)))
			model.add(MaxPooling2D((2, 2)))

			model.add(Conv2D(128, kernel_size = (3, 3), activation = "relu"))
			model.add(MaxPooling2D((2, 2)))

			model.add(Flatten())
			model.add(Dense(256, activation = 'relu'))

			model.add(Dense(10, activation = "softmax"))

			model.compile(
				optimizer = tf.keras.optimizers.Adam(
					learning_rate = 0.003,
					beta_1 = 0.8,
    			beta_2 = 0.99,
				),
				loss = "sparse_categorical_crossentropy",
				metrics = ["sparse_categorical_accuracy"]
			)

			model.fit(x_train, y_train, epochs = 10, use_multiprocessing = True, workers = 24)

			model.evaluate(x_test, y_test)
			model.save(separate_file_name)

		model = tf.keras.models.load_model(separate_file_name)
		results.append(model)

		converter = tf.lite.TFLiteConverter.from_keras_model(model)
		tflite_model = converter.convert()
		open(f"model.tflite.{i}" , "wb") .write(tflite_model)

		(_, _), (x_test, y_test) = mnist.load_data()
		(x_test, y_test) = crop_set(x_test, y_test, range(i, i + 1), debug_level)

		model_predictions = model.predict(x_test)
		for i in range(0, len(model_predictions)):
			prediction = np.argmax(model_predictions[i])
			if i >= len(predictions_results):
				predictions_results.append([])
			predictions_results[i].append(prediction)

	correct = 0
	incorrect = 0
	for i in range(0, len(predictions_results)):
		prediction = np.argmax(np.bincount(predictions_results[i]))

		if prediction == correct_results[i]:
			correct += 1
		else:
			incorrect += 1

	accuracy = correct / (correct + incorrect)
	print(f"Consensus accuracy: {accuracy}")

	return models

def load_model(file_name, debug_level = 0):
	if not os.path.isfile(file_name):
		(x_train, y_train), (x_test, y_test) = mnist.load_data()

		(x_train, y_train) = crop_set(x_train, y_train, range(0, 4), debug_level)
		(x_test, y_test) = crop_set(x_test, y_test, range(0, 4), debug_level)

		image_size = 14

		x_train = x_train / 255
		x_test = x_test / 255
		x_train = x_train.reshape(-1, image_size, image_size, 1)
		x_test = x_test.reshape(-1, image_size, image_size, 1)

		model = models.Sequential()

		model.add(Conv2D(64, kernel_size = (7, 7), activation = "relu", input_shape = (image_size, image_size, 1), padding = "same"))
		model.add(BatchNormalization(axis = 3))
		model.add(Activation('relu'))
		model.add(MaxPooling2D((2, 2)))
		model.add(Dropout(0.3))

		model.add(Conv2D(128, kernel_size = (7, 7), activation = "relu", padding = "same"))
		model.add(BatchNormalization(axis = 3))
		model.add(Activation('relu'))
		model.add(MaxPooling2D((2, 2)))
		model.add(Dropout(0.3))

		model.add(Conv2D(128, kernel_size = (7, 7), activation = "relu", padding = "same"))
		model.add(BatchNormalization(axis = 3))
		model.add(Activation('relu'))
		model.add(MaxPooling2D((2, 2)))
		model.add(Dropout(0.3))

		model.add(Flatten())
		model.add(Dense(256, activation = 'relu'))
		model.add(Dropout(0.4))

		model.add(Dense(10, activation = "softmax"))

		model.compile(
			optimizer = "adam",
			loss = "sparse_categorical_crossentropy",
			metrics = ["sparse_categorical_accuracy"]
		)

		model.fit(x_train, y_train, epochs = 20, use_multiprocessing = True, workers = 24)

		model.evaluate(x_test, y_test)
		model.save(file_name)

	model = tf.keras.models.load_model(file_name)

	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	tflite_model = converter.convert()
	open("model.tflite" , "wb") .write(tflite_model)

	if debug_level >= 1:
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
		(x_test, y_test) = crop_set(x_test, y_test, range(0, 4), debug_level)

		x_test = x_test / 255
		x_test = x_test.reshape(-1, 14, 14, 1)

		test_consensus_accuracy(model, x_test, y_test)

	return model

def test_model_against_file():
	interpreter = tf.lite.Interpreter(model_path = "model.tflite")
	interpreter.allocate_tensors()
	print(interpreter.get_signature_list())
	tflite_model = interpreter.get_signature_runner()

	image = load_image("../2-full.png")
	images = []
	images.append(crop_image(image, 0))
	images.append(crop_image(image, 1))
	images.append(crop_image(image, 2))
	images.append(crop_image(image, 3))

	for i in images:
		print_image(i, divisor = 255)

	x_test = np.asarray(images, dtype = np.float32).reshape(-1, 14, 14, 1)
	y_test = [2]
	test_consensus_accuracy(model, x_test, y_test)

	predictions = tflite_model(conv2d_input = x_test)["dense_1"]
	consensus = np.argmax(
		np.bincount(np.argmax(
				predictions,
				axis = 1
		))
	)

	print(f"tflite consensus: {consensus}")

model = load_model("model.h5", 1)
# models = load_separate_models("model.h5", 1)

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
