# conda install -c anaconda flask
# conda install -c conda-forge eventlet 
# conda install -c conda-forge python-socketio
# conda install -c conda-forge python-engineio=3.0.0
# conda install -c conda-forge tensorflow
# conda install -c conda-forge pillow
# conda install -c conda-forge numpy
# conda install -c conda-forge opencv
# pip install --upgrade tensorflow and pip install --upgrade keras


from flask import Flask
import socketio
import eventlet
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image 
import numpy as np
import cv2

sio = socketio.Server()

app = Flask(__name__) # '__main__'

speed_limit = 10

#@app.route("/home")
#def greeting():
#	return "Welcome"


def img_preprocess(img):
	img = img[60:135,:,:]
	img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
	img = cv2.GaussianBlur(img, (3,3), 0)
	img = cv2.resize(img, (200,66))
	img = img  / 255
	return img


@sio.on("telemetry")
def telemetry(sid, data):

	speed = float(data["speed"])
	image = Image.open(BytesIO(base64.b64decode(data["image"])))
	image = np.asarray(image)
	image = img_preprocess(image)
	image = np.array([image])
	steering_angle = float(model.predict(image))
	#send_control(steering_angle,1.0)
	throttle = 1.0 - speed / speed_limit
	print("Steering Angle : {}, Throttle : {} , Speed : {}".format(round(steering_angle,4),round(throttle,4),round(speed,4)))
	send_control(steering_angle,throttle)


@sio.on('connect') # message, disconnet
def connect(sid, environ):
	print("Connected")
	send_control(0, 0)

def send_control(steering_angle, throttle):
	sio.emit("steer",data = {

			"steering_angle" : steering_angle.__str__(),
			"throttle": throttle.__str__()

		})

if __name__ == "__main__":

	model = load_model("last_model.h5")
	app = socketio.Middleware(sio, app)
	eventlet.wsgi.server(eventlet.listen(('',4567)), app)














