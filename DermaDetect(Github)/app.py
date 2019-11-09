from __future__ import division, print_function
import sys
import os
import glob
import re
from pathlib import Path
from io import BytesIO
import base64
import requests
import pyrebase

# Import fast.ai Library
from fastai import *
from fastai.vision import *

# Flask utils
from flask import Flask, redirect, url_for, render_template, request
from PIL import Image as PILImage



# Define a flask app
app = Flask(__name__)
Config = {
    "apiKey": 'apiKey',
    "authDomain": 'authDomain',
    "databaseURL": 'databaseURL',
    "projectId": 'projectId',
    "storageBucket": 'storageBucket',
    "messagingSenderId": 'messagingSenderId',
    "appId": 'appId',
    "measurementId": 'measurementId'
  }

firebase = pyrebase.initialize_app(Config)

auth = firebase.auth()



NAME_OF_FILE = 'model_best' # Name of your exported file
PATH_TO_MODELS_DIR = Path('') # by default just use /models in root dir
classes = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis',
           'Dermatofibroma', 'Melanocytic nevi', 'Melanoma', 'Vascular lesions']

def setup_model_pth(path_to_pth_file, learner_name_to_load, classes):
    data = ImageDataBunch.single_from_classes(
        path_to_pth_file, classes, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
    learn = cnn_learner(data, models.densenet169, model_dir='models')
    learn.load(learner_name_to_load, device=torch.device('cpu'))
    return learn

learn = setup_model_pth(PATH_TO_MODELS_DIR, NAME_OF_FILE, classes)

def encode(img):
    img = (image2np(img.data) * 255).astype('uint8')
    pil_img = PILImage.fromarray(img)
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

def model_predict(img):
    img = open_image(BytesIO(img))
    pred_class,pred_idx,outputs = learn.predict(img)
    formatted_outputs = ["{:.1f}%".format(value) for value in [x * 100 for x in torch.nn.functional.softmax(outputs, dim=0)]]
    pred_probs = sorted(
            zip(learn.data.classes, map(str, formatted_outputs)),
            key=lambda p: p[1],
            reverse=True
        )

    img_data = encode(img)
    result = {"class":pred_class, "probs":pred_probs, "image":img_data}
    return render_template('result.html', result=result)


@app.route('/', methods=['GET', "POST"])
def index():
# def basic():
    unsuccessful = 'Please check your credentials'
    successful = 'Login successful'
    Successful_Registration = 'You are registered and can log in now'
    Successful_Password = 'Your Password was reset'
    Unsuccessful_Password = 'Please enter only your email adress and hit the button again'
    full_filename = '/templates/img1.jpg'

    if request.method == 'POST':
        email = request.form['name']
        password = request.form['pass']
        option = request.form['product']
        # print('new:',option)
        if option == 'Sign in':
            try:
                auth.sign_in_with_email_and_password(email,password)
                return render_template('index.html')
            except:
                return render_template('new.html', us=unsuccessful, user_image = full_filename)
        elif option == 'Sign up':
            try:
                auth.create_user_with_email_and_password(email,password)
                return render_template('new.html', sr=Successful_Registration, user_image = full_filename)
            except:
                return render_template('new.html', sr=unsuccessful, user_image = full_filename)
        elif option == 'Forgot':
            try:
                auth.send_password_reset_email(email)
                return render_template('new.html', sp=Successful_Password, user_image = full_filename)
            except:
                return render_template('new.html', up=Unsuccessful_Password, user_image = full_filename)


    return render_template('new.html', user_image = full_filename)

    # Main page
    #return render_template('index.html')
@app.route('/signup', methods=['GET', "POST"])
def register():
    if request.method == 'POST':
        email = request.form['name']
        password = request.form['pass']
        auth.create_user_with_email_and_password(email, password)
    return render_template('new.html')



@app.route('/upload', methods=["POST", "GET"])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        img = request.files['file'].read()
        if img != None:
        # Make prediction
            preds = model_predict(img)
            return preds
    return 'OK'

@app.route("/classify-url", methods=["POST", "GET"])
def classify_url():
    if request.method == 'POST':
        url = request.form["url"]
        if url != None:
            response = requests.get(url)
            preds = model_predict(response.content)
            return preds
    return 'OK'


if __name__ == '__main__':
    port = os.environ.get('PORT', 8008)

    if "prepare" not in sys.argv:
        app.run(debug=False, host='0.0.0.0', port=port)
