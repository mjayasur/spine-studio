import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory, jsonify
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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

vertebrae_labels = [
    {"value": 27, "label": "vertebrae_L5"},
    {"value": 28, "label": "vertebrae_L4"},
    {"value": 29, "label": "vertebrae_L3"},
    {"value": 30, "label": "vertebrae_L2"},
    {"value": 31, "label": "vertebrae_L1"},
    {"value": 32, "label": "vertebrae_T12"},
    {"value": 33, "label": "vertebrae_T11"},
    {"value": 34, "label": "vertebrae_T10"},
    {"value": 35, "label": "vertebrae_T9"},
    {"value": 36, "label": "vertebrae_T8"},
    {"value": 37, "label": "vertebrae_T7"},
    {"value": 38, "label": "vertebrae_T6"},
    {"value": 39, "label": "vertebrae_T5"},
    {"value": 40, "label": "vertebrae_T4"},
    {"value": 41, "label": "vertebrae_T3"},
    {"value": 42, "label": "vertebrae_T2"},
    {"value": 43, "label": "vertebrae_T1"},
    {"value": 44, "label": "vertebrae_C7"},
    {"value": 45, "label": "vertebrae_C6"},
    {"value": 46, "label": "vertebrae_C5"},
    {"value": 47, "label": "vertebrae_C4"},
    {"value": 48, "label": "vertebrae_C3"},
    {"value": 49, "label": "vertebrae_C2"},
    {"value": 50, "label": "vertebrae_C1"},
]

def allowed_file(filename):
    return True
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'nii', 'nii.gz'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        input_img = nib.load(filepath)
        mask_nib = totalsegmentator(input_img)
        mask_data = mask_nib.get_fdata()
        present_vertebrae = [v for v in vertebrae_labels if np.any(mask_data == v['value'])]

        return jsonify(present_vertebrae)
    return redirect(request.url)

@app.route('/process', methods=['POST'])
def process_vertebrae():
    selected_vertebrae = request.form.getlist('vertebrae')
    selected_vertebrae = [int(v) for v in selected_vertebrae]

    file = os.listdir(app.config['UPLOAD_FOLDER'])[0]
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file)
    input_img = nib.load(filepath)
    mask_nib = totalsegmentator(input_img)
    mask_data = mask_nib.get_fdata()

    results = []
    for vertebra_value in selected_vertebrae:
        largest_component_mask, z_min, z_max = find_l1_vert(mask_nib, vertebra_value)
        fig, avg_hu = plot_top_slices(filepath, largest_component_mask, z_min, z_max, overlay_mask=True)
        img = BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        img_b64 = base64.b64encode(img.getvalue()).decode('utf8')
        img.close()
        results.append({"filename": f"data:image/png;base64,{img_b64}", "avg_hu": avg_hu})
    
    return jsonify(results)

def find_l1_vert(mask_nii, mask_value=30):
    mask_data = mask_nii.get_fdata()
    l1_mask = mask_data == mask_value
    labeled_array, num_features = label(l1_mask)
    largest_component = 0
    largest_size = 0
    for i in range(1, num_features + 1):
        component_size = np.sum(labeled_array == i)
        if component_size > largest_size:
            largest_size = component_size
            largest_component = i
    largest_component_mask = labeled_array == largest_component
    l1_coords = np.argwhere(largest_component_mask)
    z_min, y_min, x_min = l1_coords.min(axis=0)
    z_max, y_max, x_max = l1_coords.max(axis=0)
    return largest_component_mask, z_min, z_max

def kadane(arr):
    max_sum = float('-inf')
    current_sum = 0
    start = end = s = 0
    for i in range(len(arr)):
        current_sum += arr[i]
        if current_sum > max_sum:
            max_sum = current_sum
            start = s
            end = i
        if current_sum < 0:
            current_sum = 0
            s = i + 1
    return max_sum, start, end

def find_largest_rectangle(matrix):
    rows, cols = matrix.shape
    max_sum = float('-inf')
    final_left = final_right = final_top = final_bottom = 0
    for left in range(cols):
        temp = np.zeros(rows)
        for right in range(left, cols):
            temp += matrix[:, right]
            current_sum, start, end = kadane(temp)
            if current_sum > max_sum:
                max_sum = current_sum
                final_left = left
                final_right = right
                final_top = start
                final_bottom = end
    return final_top, final_left, final_bottom, final_right

def plot_top_slices(image_path, mask, z_min, z_max, overlay_mask=False):
    image_nii = nib.load(image_path)
    image_data = image_nii.get_fdata()
    slice_sums = [(z, np.sum(mask[z, :, :])) for z in range(z_min, z_max + 1)]
    slice_sums.sort(key=lambda x: x[1], reverse=True)
    top_slices = slice_sums[:5]
    fig, axs = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle('Top 5 Axial Slices with Largest Mask Areas', fontsize=16)
    average_hu_per_slice = []
    for i, (z, _) in enumerate(top_slices):
        axial_slice = image_data[z, :, :]
        mask_slice = mask[z, :, :]
        mask_slice_converted = np.where(mask_slice, 1, -1000)
        axs[i].imshow(axial_slice, cmap='gray', origin='lower')
        if overlay_mask:
            overlay = np.zeros_like(axial_slice)
            overlay[mask_slice] = 1
            axs[i].imshow(np.ma.masked_where(overlay == 0, overlay), cmap='jet', alpha=0.5, origin='lower')
        max_rect = find_largest_rectangle(mask_slice_converted)
        (row1, col1, row2, col2) = max_rect
        rect_width = col2 - col1
        rect_height = row2 - row1
        ellipse_center = ((col1 + col2) / 2, (row1 + row2) / 2)
        ellipse_width = rect_width
        ellipse_height = rect_height
        axs[i].add_patch(Rectangle((col1, row1), rect_width, rect_height, edgecolor='red', facecolor='none'))
        axs[i].add_patch(Ellipse(ellipse_center, ellipse_width, ellipse_height, edgecolor='blue', facecolor='none'))
        rect_mask = mask_slice[row1:row2+1, col1:col2+1]
        rect_values = axial_slice[row1:row2+1, col1:col2+1][rect_mask]
        average_hu = np.mean(rect_values)
        average_hu_per_slice.append(average_hu)
        axs[i].set_title(f'Slice {z}\nAvg HU: {average_hu:.2f}')
        axs[i].axis('off')
    overall_average_hu = np.mean(average_hu_per_slice)
    return fig, overall_average_hu

if __name__ == '__main__':
    app.run(debug=True)
