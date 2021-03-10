import cv2
import tensorflow as tf 
import os
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np

cascade_path =  "/Users/kimotoakirasuke/Documents/Aidemy_kuso/TXT_member/txt_webpage/haarcascade_xml/haarcascade_frontalface_default.xml"
member_name = ["カン テヒョン", "スビン", "ヨン ジュン", "ヒョニんカイ", "ボムギュ"]
UPLOAD_FOLDER = os.curdir
model_path = os.path.join(os.getcwd(), "my_model_3")

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == "":
            flash('ファイルがありません')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            img = cv2.imread(filepath)
            img = cv2.resize(img, (300, 300))
            img = img[np.newaxis]
            img = tf.convert_to_tensor(img, np.float32)
            img /= 255
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            model = load_model(os.path.join(model_path, 'my_model_03.h5'), compile=False)
            pred = model.predict(img)
            ans = member_name[np.argmax(pred)] + "です"

            return render_template("./txt.html", answer=ans)
    return render_template("./txt.html", answer="")

if  __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)

def judge_wehere_face(path):
    img = cv2.imread(path)
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cascade_path)
    image_gray = cv2.resize(image_gray, (300, 300))
    foreacst = cascade.detectMultiScale(image_gray, scaleFactor=1.1)
    return foreacst

def distinguish_img(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (300, 300))
    img_h, img_w = img.shape[0], img.shape[1]
    return img

def convert_tensor(img):
    img = cv2.resize(img, (300, 300))
    img = img[np.newaxis]
    img = tf.convert_to_tensor(img, np.float32)
    img /= 255
    return img


def model(image):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    model = load_model('./my_model_03.h5', compile=False)
    pred = model.predict(image)
    ans = member_name[np.argmax(pred)] + "です"
    return ans





        

