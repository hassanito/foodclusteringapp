from flask import Flask, request, make_response, render_template, redirect, url_for, send_file,jsonify
import keras
import numpy as np
import os
from PIL import Image
from io import BytesIO, StringIO
from zipfile import ZipFile

from werkzeug.utils import secure_filename
import DataHandler
import threading
import ML
ALLOWED_FILE_EXTENSIONS = {'txt'}
os.chdir('C:\\Users\\hassanelhajj\\desktop\\docs2\\FypApi\\foodclusteringapp')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_FILE_EXTENSIONS


app = Flask(__name__)
label = 'img'

#set paths to upload folder
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] =os.path.join(APP_ROOT, 'ProductsLists')
app.config['PREDICT_IMAGE'] =os.path.join(APP_ROOT,'static')

@app.route('/uploadlist', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return "no file"
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return "no selected file"
        if file and allowed_file(file.filename):
            uniqueFileName = DataHandler.GetFileName(app.config['UPLOAD_FOLDER'])
            filename = secure_filename(uniqueFileName)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            productsFile = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            DataHandler.MergeCheck(app.config['UPLOAD_FOLDER'])
            task = threading.Thread(target=ML.TrainServer, args=(10,))
            task.daemon = True
            task.start()
            return "Products list uploaded and merged"
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/predictimage',methods=['GET','POST'])
def PredictImage():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return "no file"
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return "no selected file"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['PREDICT_IMAGE'], filename))
            """task = threading.Thread(target=DataHandler.Train, args=(10,))
            task.daemon = True
            task.start()"""
            prediction = "5"
            return "Pediction = "+prediction
app.run()