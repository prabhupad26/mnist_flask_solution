from flask import Flask, flash, redirect, render_template, request, session, abort
from time import strftime
import base64
import json
from predict import DigitRecognition

app = Flask(__name__)


# app.debug = True
@app.route('/')
def hello_world():
    return 'Hello World'


@app.route('/home')
def index():
    return render_template('home_page.html')


@app.route('/predict_image', methods=['POST'])
def predict_image():
    input_img = str(request.data)
    input_img_filtered = input_img.split(',')[1]
    input_img_encoded = bytes(input_img_filtered, 'utf-8')
    input_img_decoded = base64.decodebytes(input_img_encoded)
    target_img = 'static/uploads/sample{0}.png'.format(strftime('%Y%m%d%H%M%S'))
    with open(target_img, 'wb') as file:
        file.write(input_img_decoded)
    model_obj = DigitRecognition(target_img, model_arch='CNN')
    predicted_output = model_obj.predict()
    return predicted_output


if __name__ == '__main__':
    app.run()
