import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from flask import Flask, request, render_template, redirect, flash

app = Flask(__name__)
app.secret_key = "abcdef123"

IMG_SIZE = 96
CATEGORIES = ['CH07', 'CH30', 'Corona', 'Dråben', 'Myren', 'Oxford', 'Svanen', 'Trinidad', 'Y']


@app.route("/", methods=("GET", "POST"))
def index():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        image = request.files['file']
        if image.filename == '':
            flash('No file selected')
            return redirect(request.url)
        if image:
            image.save(image.filename)
            result = categorize_image(image.filename)
            return render_template("result.html", result=result)
    return render_template("index.html")

def categorize_image(image):
    image = format_image(image)
    model = keras.models.load_model("model")
    prediction = model.predict(image)
    percentages = {label: "{:.5f}".format(prediction[0][CATEGORIES.index(label)].item()) for label in CATEGORIES}
    return percentages

def format_image(filepath):
    # np_array = np.frombuffer(image.read(1024))
    # decode_image = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)
    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    image = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
    image = image.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    # image = np.reshape(image, (1, IMG_SIZE,IMG_SIZE,3))
    os.remove(filepath)
    return image
