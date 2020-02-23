import os
import sys


from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import numpy as np
from util import base64_to_pil

app = Flask(__name__)

from keras.applications.resnet50 import ResNet50
model = ResNet50(weights='imagenet')

print('Model loaded. Check http://127.0.0.1:5000/')

MODEL_PATH = 'models/your_model.h5'

def model_predict(img, model):
    img = img.resize((224, 224))

    
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x, mode='tf')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        img = base64_to_pil(request.json)

        preds = model_predict(img, model)

        pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode

        result = str(pred_class[0][0][1])               # Convert to string
        result = result.replace('_', ' ').capitalize()
       
        return jsonify(result=result, probability=pred_proba)

    return None


if __name__ == '__main__':
    
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
