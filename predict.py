import base64
import numpy as np
import io
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from flask import request
from flask import jsonify
from flask import Flask
from resnet import ResnetBlock

app = Flask(__name__)

pred_dict = {'0': 'BENIGN',
            '1': 'MALIGNANT',
            '2': 'BENIGN_WITHOUT_CALLBACK'}

model  = load_model('ResNet50.h5', custom_objects={'Functional':ResnetBlock})

def preprocess_image(image, width=255, height=255, interpolation=cv2.INTER_AREA):
    try:
        image = cv2.resize(image, (width, height), interpolation = interpolation)
        b , g, r = cv2.split(image)
        image = cv2.merge((r,g,b))
        return image
    except Exception:
        pass

@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    # open the image - not sure how the image is received
    image = Image.open(io.BytesIO(decoded))
    # use this or 
    # image = cv2.imread(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(255, 255))
    processed_image = np.expand_dims(processed_image, axis=0)
    prediction = model.predict(processed_image).tolist()
    
    response = {
        'prediction': pred_dict[str(np.argmax(prediction))],
        'confidence': np.max(prediction)
    }
    return jsonify(response)