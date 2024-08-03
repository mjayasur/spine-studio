import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from matplotlib.patches import Ellipse
from matplotlib.colors import ListedColormap

import sys
import nibabel as nib
import uuid
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
NIB_FOLDER = 'nib_files'
IMAGE_FOLDER = 'static/images'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['NIB_FOLDER'] = NIB_FOLDER
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

vertebrae_dict = {
    26: "vertebrae_S1",
    27: "vertebrae_L5",
    28: "vertebrae_L4",
    29: "vertebrae_L3",
    30: "vertebrae_L2",
    31: "vertebrae_L1",
    32: "vertebrae_T12",
    33: "vertebrae_T11",
    34: "vertebrae_T10",
    35: "vertebrae_T9",
    36: "vertebrae_T8",
    37: "vertebrae_T7",
    38: "vertebrae_T6",
    39: "vertebrae_T5",
    40: "vertebrae_T4",
    41: "vertebrae_T3",
    42: "vertebrae_T2",
    43: "vertebrae_T1",
    44: "vertebrae_C7",
    45: "vertebrae_C6",
    46: "vertebrae_C5",
    47: "vertebrae_C4",
    48: "vertebrae_C3",
    49: "vertebrae_C2",
    50: "vertebrae_C1"
}


@app.route('/upload-file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify(success=False, message="No file part")
    files = request.files.getlist('file')
    if not files:
        return jsonify(success=False, message="No selected file")
    
    output_uuid = str(uuid.uuid4())
    available_verts = []

    for file in files:
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            input_nib = nib.load(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            output_nib = totalsegmentator(input_nib)
            nib.save(input_nib, os.path.join(app.config['NIB_FOLDER'], f"{output_uuid}.nii.gz"))
            nib.save(output_nib, os.path.join(app.config['NIB_FOLDER'], f"{output_uuid}_mask.nii.gz"))
            
            for val in np.unique(output_nib.get_fdata()):
                if val in vertebrae_dict:
                    available_verts.append(vertebrae_dict[val].split("_")[1])
            print (output_uuid)
            print (available_verts)
    
    return jsonify(success=True, message="Files successfully uploaded", uuid=output_uuid, vertebrae=sorted(available_verts))

def load_nii(filepath):
    return nib.load(filepath).get_fdata()

def find_largest_connected_component(mask_data):
    labeled, num_features = label(mask_data, return_num=True)
    largest_component = 0
    max_size = 0
    for i in range(1, num_features + 1):
        size = np.sum(labeled == i)
        if size > max_size:
            max_size = size
            largest_component = i
    return (labeled == largest_component).astype(int)

def kadanes_algorithm(arr):
    max_sum = 0
    left, right, up, bottom = -1, -1, -1, -1
    temp = np.zeros(len(arr))
    for l in range(len(arr[0])):
        temp[:] = 0
        for r in range(l, len(arr[0])):
            temp += arr[:, r]
            current_sum = 0
            start = 0
            for i in range(len(arr)):
                current_sum += temp[i]
                if current_sum > max_sum:
                    max_sum = current_sum
                    left, right, up, bottom = l, r, start, i
                if current_sum < 0:
                    current_sum = 0
                    start = i + 1
    return left, right, up, bottom

def fit_ellipse_to_rectangle(rect_coords, image_slice):
    left, right, up, bottom = rect_coords
    center_x = (left + right) / 2
    center_y = (up + bottom) / 2
    axis_a = (right - left) / 2 - 2  # Semi-major axis
    axis_b = (bottom - up) / 2 - 2  # Semi-minor axis
    
    # Create an ellipse patch
    ellipse = Ellipse((center_x, center_y), width=axis_a*2, height=axis_b*2, edgecolor='red', facecolor='none')
                
    # Create a grid of the same size as the image
    y_indices, x_indices = np.ogrid[:image_slice.shape[0], :image_slice.shape[1]]
    
    # Check which pixels fall within the ellipse
    elliptical_mask = ((x_indices - center_x) / axis_a) ** 2 + ((y_indices - center_y) / axis_b) ** 2 <= 1
    
    # Calculate the mean of the pixels within the ellipse
    average_value = np.mean(image_slice[elliptical_mask])
    
    return ellipse, average_value

@app.route('/calculate-hounsfield', methods=['GET'])
def calculate_hounsfield():
    id = request.args.get('id')
    modality = request.args.get('modality')
    vertebrae = request.args.get('vertebrae')

    if not id or not modality or not vertebrae:
        return jsonify(success=False, message="Missing parameters"), 400

    try:
        # Map the vertebrae string to the corresponding ID
        vertebrae_id = next((key for key, value in vertebrae_dict.items() if value.split('_')[1] == vertebrae), None)
        if vertebrae_id is None:
            return jsonify(success=False, message="Invalid vertebrae level"), 400

        # Load the NIfTI file using the id
        input_nib_path = os.path.join(app.config['NIB_FOLDER'], f"{id}.nii.gz")
        mask_nib_path = os.path.join(app.config['NIB_FOLDER'], f"{id}_mask.nii.gz")

        if not os.path.exists(input_nib_path) or not os.path.exists(mask_nib_path):
            return jsonify(success=False, message="NIfTI files not found"), 404

        input_nib = nib.load(input_nib_path)
        mask_nib = nib.load(mask_nib_path)

        input_data = input_nib.get_fdata()
        mask_data = mask_nib.get_fdata()

        # Get the axial slice index based on the vertebrae ID
        axial_slice_indices = np.where(mask_data == vertebrae_id)[1]
        if len(axial_slice_indices) == 0:
            return jsonify(success=False, message="Vertebrae level not found in the mask"), 404

        axial_slice_index = axial_slice_indices[len(axial_slice_indices) // 2]

        # Extract the axial slice for the input and mask
        axial_slice = input_data[:, axial_slice_index, :]
        mask_slice = np.where(mask_data[:, axial_slice_index, :] == vertebrae_id, 1, -2).astype(np.int8)

        # Apply Kadane's algorithm to find the largest rectangle
        rect_coords = kadanes_algorithm(mask_slice)
        ellipse, average_hu = fit_ellipse_to_rectangle(rect_coords, axial_slice)

        # Get the midsagittal slice index
        midsagittal_index = input_data.shape[2] // 2

        # Extract the midsagittal slice for the input and mask
        midsagittal_slice = input_data[:,  :,midsagittal_index]
        mask_sagittal_slice = (mask_data[:,  :, midsagittal_index] == vertebrae_id).astype(np.uint8)

        # Rotate the midsagittal slice by 90 degrees
        midsagittal_slice_rotated = np.rot90(midsagittal_slice)
        mask_sagittal_slice_rotated = np.rot90(mask_sagittal_slice)

        # Create unique filenames for the resulting images
        output_uuid = str(uuid.uuid4())
        axial_slice_path = os.path.join(app.config['IMAGE_FOLDER'], f"{output_uuid}_axial.png")
        roi_placement_path = os.path.join(app.config['IMAGE_FOLDER'], f"{output_uuid}_roi.png")
        sagittal_slice_path = os.path.join(app.config['IMAGE_FOLDER'], f"{output_uuid}_sagittal.png")

        # Save the axial slice image
        plt.imsave(axial_slice_path, axial_slice, cmap='gray')

        # Overlay the ellipse on the axial slice
        fig, ax = plt.subplots()
        ax.imshow(axial_slice, cmap='gray')
        ax.add_patch(ellipse)
        plt.axis('off')
        plt.savefig(roi_placement_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Save the rotated midsagittal slice with mask overlay
        fig, ax = plt.subplots()
        ax.imshow(midsagittal_slice_rotated, cmap='gray')
        ax.imshow(mask_sagittal_slice_rotated, cmap='jet', alpha=0.5)
        plt.axis('off')
        plt.savefig(sagittal_slice_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        return jsonify(
            success=True,
            axialSliceUrl=f"/static/images/{output_uuid}_axial.png",
            roiPlacementUrl=f"/static/images/{output_uuid}_roi.png",
            sagittalSliceUrl=f"/static/images/{output_uuid}_sagittal.png",
            averageHU=average_hu
        )

    except Exception as e:
        return jsonify(success=False, message=str(e)), 500

@app.route('/calculate-sarcopenia', methods=['GET'])
def calculate_sarcopenia():
    id = request.args.get('id')
    modality = request.args.get('modality')

    if not id or not modality:
        return jsonify(success=False, message="Missing parameters"), 400

    try:
        # Load the NIfTI file using the id
        input_nib_path = os.path.join(app.config['NIB_FOLDER'], f"{id}.nii.gz")
        mask_nib_path = os.path.join(app.config['NIB_FOLDER'], f"{id}_mask.nii.gz")

        if not os.path.exists(input_nib_path) or not os.path.exists(mask_nib_path):
            return jsonify(success=False, message="NIfTI files not found"), 404

        input_nib = nib.load(input_nib_path)
        mask_nib = nib.load(mask_nib_path)

        input_data = input_nib.get_fdata()
        mask_data = mask_nib.get_fdata()

        # Get the axial slice index for L3
        l3_indices = np.where(mask_data == 29)[1]
        if len(l3_indices) == 0:
            return jsonify(success=False, message="L3 level not found in the mask"), 404

        l3_slice_index = l3_indices[len(l3_indices) // 2]

        # Extract the axial slice for the input and mask
        axial_slice = input_data[:, l3_slice_index, :]
        mask_slice = mask_data[:, l3_slice_index, :]

        # Calculate iliopsoas and autochthon areas
        iliopsoas_area = np.sum((mask_slice == 88) | (mask_slice == 89))
        autochthon_area = np.sum((mask_slice == 86) | (mask_slice == 87))

        iliopsoas_area_cm2 = iliopsoas_area * np.prod(input_nib.header.get_zooms()[:2]) / 100
        autochthon_area_cm2 = autochthon_area * np.prod(input_nib.header.get_zooms()[:2]) / 100

        # Create the binary mask
        binary_mask = np.zeros_like(mask_slice)
        binary_mask[mask_slice == 88] = 1
        binary_mask[mask_slice == 89] = 1
        binary_mask[mask_slice == 86] = 2
        binary_mask[mask_slice == 87] = 2

        # Create a custom colormap with transparency
        cmap = ListedColormap(['black', 'red', 'blue'])
        norm = plt.Normalize(vmin=0, vmax=2)

        # Create unique filenames for the resulting images
        sarcopenia_mask_path = os.path.join(app.config['IMAGE_FOLDER'], f"{id}_sarcopenia_mask.png")

        # Display the image and the binary mask
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(axial_slice, cmap='gray')
        mask_overlay = ax.imshow(binary_mask, cmap=cmap, norm=norm, alpha=0.3)
        ax.axis('off')
        plt.savefig(sarcopenia_mask_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        return jsonify(
            success=True,
            maskUrl=f"/static/images/{id}_sarcopenia_mask.png",
            iliopsoasArea=iliopsoas_area_cm2,
            autochthonArea=autochthon_area_cm2
        )

    except Exception as e:
        return jsonify(success=False, message=str(e)), 500
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
    app.run(host='0.0.0.0', port=sys.argv[1], debug=True)
