import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory, jsonify
from werkzeug.utils import secure_filename

import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import label
from totalsegmentator.python_api import totalsegmentator
import base64
from io import BytesIO
from matplotlib.patches import Rectangle, Ellipse
# Configure upload folder
UPLOAD_FOLDER = 'uploads'
# Determine the root path based on the execution context
if '__file__' in globals():
    root_path = os.path.dirname(os.path.abspath(__file__))
else:
    root_path = os.getcwd()

app = Flask(__name__, root_path=root_path)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/upload-file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify(success=False, message="No file part")
    files = request.files.getlist('file')
    if not files:
        return jsonify(success=False, message="No selected file")
    
    for file in files:
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            input_nib = nib.load(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            output_nib = totalsegmentator(input_nib)
    
    return jsonify(success=True, message="Files successfully uploaded")

@app.route('/output_xr')
def output_xr():
    return render_template('output_xr.html')

@app.route('/output_ct')
def output_ct():
    return render_template('output_ct.html')

@app.route('/output_mri')
def output_mri():
    return render_template('output_mri.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/users')
def users():
    return render_template('users.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/get_started')
def get_started():
    return render_template('get_started.html')

@app.route('/loading')
def loading():
    return render_template('loading.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
