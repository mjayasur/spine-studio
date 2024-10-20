import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from matplotlib.patches import Ellipse
from matplotlib.colors import ListedColormap
from PIL import Image
import sys
import nibabel as nib
from PIL import ImageEnhance
import uuid
import numpy as np
import matplotlib
from scipy.ndimage import *
import cv2
matplotlib.use('Agg')
from ultralytics import YOLO
# Load the YOLO model for X-ray images
# Replace the path with the actual path to your trained YOLO model weights
xr_model = YOLO("last.pt")  # Path to your last saved weights
mr_model = YOLO("best.pt")  # Path to your last saved weights

import matplotlib.pyplot as plt
from scipy.ndimage import label
from totalsegmentator.python_api import totalsegmentator
import base64
from io import BytesIO
import io
import csv
import requests
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
# Helper function to check if two points are close
def _is_close(point1, point2, threshold=5):
    return abs(point1[0] - point2[0]) <= threshold and abs(point1[1] - point2[1]) <= threshold

def calculate_vbq_roi(image_path):
    results = mr_model(image_path)
    all_annotations = {}

    for i, result in enumerate(results):
        im_bgr = result.plot()  # BGR-order numpy array
        im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

        # Increase brightness
        enhancer = ImageEnhance.Brightness(im_rgb)
        im_bright = enhancer.enhance(3)  # 3 is the factor by which brightness is increased. Adjust as needed.

        # Convert back to numpy array for plotting
        im_bright_np = np.array(im_bright)

        # Get bounding boxes and classes
        bboxes, classes = result.boxes.xyxy, result.boxes.cls

        # Extract centers and sort by y value
        centers = [(int((x1 + x2) / 2), int((y1 + y2) / 2), min((x2 - x1) // 2, (y2 - y1) // 2)) for x1, y1, x2, y2 in bboxes]
        centers_sorted = sorted(centers, key=lambda x: x[1], reverse=True)  # Sort by y value in descending order

        # Filter out close duplicates
        filtered_centers = []
        for center in centers_sorted[::-1]:
            if all(not _is_close(center, existing_center) for existing_center in filtered_centers):
                filtered_centers.insert(0, center)
        filtered_centers = filtered_centers[1:5]

        # Collecting annotations
        image_annotations = []
        j = 0
        for center_x, center_y, radius in filtered_centers[::-1]:
            image_annotations.append({
                "x": int(center_x),
                "y": int(center_y),
                "radius": int(radius),
                "annotation": f"L{j + 1}"  # Assuming all these are vertebral annotations
            })
            j += 1

        L3_idx = min(len(filtered_centers) - 1, 1)
        box_x1 = int(filtered_centers[L3_idx][0] + filtered_centers[L3_idx][2])
        box_y1 = int(filtered_centers[L3_idx][1] - 0.75 * filtered_centers[L3_idx][2])
        box_x2 = int(filtered_centers[L3_idx][0] + 2 * filtered_centers[L3_idx][2])
        box_y2 = int(filtered_centers[L3_idx][1] + 0.75 * filtered_centers[L3_idx][2])

        # Ensure indices are within the bounds of the image
        box_x1 = max(0, box_x1)
        box_y1 = max(0, box_y1)
        box_x2 = min(result.orig_img.shape[1], box_x2)
        box_y2 = min(result.orig_img.shape[0], box_y2)

        # Extract the region of interest (ROI) to the right of the L3 index
        roi = result.orig_img[box_y1:box_y2, box_x1:box_x2]

        # Convert ROI to grayscale if it's not already (assuming it might be an RGB image)
        if roi.ndim == 3 and roi.shape[2] == 3:
            roi_gray = np.mean(roi, axis=2).astype(np.float32)
        else:
            roi_gray = roi.astype(np.float32)

        # Apply a uniform filter to calculate the mean values in 7x7 regions
        mean_filter = uniform_filter(roi_gray, size=6)

        # Find the coordinates of the smallest mean value
        min_mean_y, min_mean_x = np.unravel_index(np.argmin(mean_filter), mean_filter.shape)
        csf_x = box_x1 + min_mean_x + 3  # Center of the 7x7 box
        csf_y = box_y1 + min_mean_y + 3  # Center of the 7x7 box

        # Add CSF annotation
        image_annotations.append({
            "x": int(csf_x),
            "y": int(csf_y),
            "radius": 3,
            "annotation": "CSF"
        })

        # Add image annotations to all_annotations
        all_annotations[os.path.basename(image_path)] = image_annotations

    return all_annotations

# Load image and extract pixel intensities
def load_image(filepath):
    return np.array(Image.open(filepath).convert('L'))  # Convert image to grayscale

def extract_intensity(image, annotation):
    x, y, radius = annotation['x'], annotation['y'], annotation['radius']
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask = cv2.circle(mask, (int(x), int(y)), int(radius), 1, thickness=-1)
    return image[mask == 1]

def calculate_median_intensity(image, annotations):
    intensities = {ann['annotation']: extract_intensity(image, ann) for ann in annotations}
    median_intensities = {key: np.median(val) for key, val in intensities.items()}
    return median_intensities

# Calculate VBQ score
def calculate_vbq(median_intensities):
    vertebra_medians = list(median_intensities.values())
    csf_median = median_intensities['CSF']
    vbq_score = np.median(vertebra_medians) / csf_median if csf_median != 0 else np.inf
    return vbq_score, vertebra_medians, csf_median

# Function to display and save the annotated image
def display_annotated_image(image_path, annotations, output_path, title="Annotated Image"):
    # Load the image
    im_rgb = Image.open(image_path)
    im_np = np.array(im_rgb)

    # Plot image
    plt.imshow(im_np)
    ax = plt.gca()

    # Draw circles based on annotations
    for annotation in annotations:
        center_x = annotation["x"]
        center_y = annotation["y"]
        radius = annotation["radius"]
        label = annotation["annotation"]

        # Draw circle
        circle = plt.Circle((center_x, center_y), radius, color='red', fill=False, linewidth=2)
        ax.add_patch(circle)

        # Add label
        plt.text(center_x, center_y, label, color='yellow', fontsize=8, ha='center', va='center')

    plt.title(title)
    plt.axis('off')  # Hide axes

    # Save the plot to a file
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

@app.route('/upload-file', methods=['POST'])
def upload_file():
    if 'file' not in request.files or 'modality' not in request.form:
        return jsonify(success=False, message="Missing file or modality")
    
    files = request.files.getlist('file')
    modality = request.form['modality']  # 'XR' for X-ray, 'MR' for MRI, 'CT' for CT
    
    if not files:
        return jsonify(success=False, message="No selected file")

    output_uuid = str(uuid.uuid4())
    available_verts = []

    for file in files:
        # Handling NIfTI (CT) files
        if file and "nii" in file.filename:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            input_nib = nib.load(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            output_nib = totalsegmentator(input_nib)
            nib.save(input_nib, os.path.join(app.config['NIB_FOLDER'], f"{output_uuid}.nii.gz"))
            nib.save(output_nib, os.path.join(app.config['NIB_FOLDER'], f"{output_uuid}_mask.nii.gz"))
            modality = 'CT'
            for val in np.unique(output_nib.get_fdata()):
                if val in vertebrae_dict:
                    available_verts.append(vertebrae_dict[val].split("_")[1])
            print(output_uuid)
            print(available_verts)

        # Handling JPEG/PNG files (X-ray images)
        elif file and (".jpg" in file.filename.lower() or ".jpeg" in file.filename.lower() or ".png" in file.filename.lower()):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{output_uuid}_{filename}")
            file.save(filepath)
            print (modality)
            if modality == 'XR':  # X-ray modality
                # YOLO model can be used for vertebrae detection (L1-L5 assumed for X-ray)
                available_verts = ['L1', 'L2', 'L3', 'L4', 'L5']
                # Add any X-ray specific processing here (e.g., keypoints detection, etc.)
            
            elif modality == 'MRI':  # MRI modality (perform VBQ calculation)
                print (filepath)
                annotations = calculate_vbq_roi(filepath)
                image = load_image(filepath)
                median_intensities = calculate_median_intensity(image, annotations[os.path.basename(filepath)])
                vbq_score, vertebra_medians, csf_median = calculate_vbq(median_intensities)
                
                # Save the annotated image
                annotated_image_path = os.path.join(IMAGE_FOLDER, f"annotated_mr_{output_uuid}.png")
                print (annotated_image_path)
                display_annotated_image(filepath, annotations[os.path.basename(filepath)], annotated_image_path)

                return jsonify(
                    success=True,
                    message="VBQ successfully calculated",
                    uuid=output_uuid,
                    modality=modality,
                    vbq_score=vbq_score,
                    vertebra_medians=vertebra_medians,
                    csf_median=csf_median,
                    image_url=annotated_image_path
                )

    # Return based on modality
    if modality == 'CT':
        return jsonify(success=True, message="CT files successfully uploaded", uuid=output_uuid, modality=modality, vertebrae=sorted(available_verts))
    elif modality == 'XR':
        return jsonify(success=True, message="X-ray files successfully uploaded", uuid=output_uuid, modality=modality, vertebrae=available_verts)
    else:
        return jsonify(success=False, message="Unsupported file type")

def euclidean_distance(p1, p2):
    """Calculate the Euclidean distance between two points (x1, y1) and (x2, y2)."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def classify_height_loss(left_height, right_height):
    """Classify height loss severity based on the percentage of vertebral height loss."""
    # Determine which side is smaller (compressed)
    min_height = min(left_height, right_height)
    max_height = max(left_height, right_height)

    # Calculate the percentage of height loss
    height_loss_percentage = (max_height - min_height) / max_height * 100

    # Classify the height loss based on the thresholds
    if height_loss_percentage <= 20:
        classification = "No compression fracture"
    elif height_loss_percentage <= 25:
        classification = "Mild"
    elif 25 < height_loss_percentage <= 40:
        classification = "Moderate"
    else:
        classification = "Severe"

    return classification, height_loss_percentage

@app.route('/calculate-compression-fracture', methods=['GET'])
def calculate_compression_fracture():
    id = request.args.get('id')

    if not id:
        return jsonify(success=False, message="Missing parameters"), 400

    try:
        # Map vertebrae to indices (L1-L5 correspond to index 0-4)
        vertebrae_map = ['L1', 'L2', 'L3', 'L4', 'L5']

        # Load the uploaded image
        image_filenames = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.startswith(id)]
        if not image_filenames:
            return jsonify(success=False, message="Image not found"), 404

        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filenames[0])
        original_image = Image.open(image_path)

        # Perform inference to get the predicted keypoints
        results = xr_model.predict(source=image_path, save=False)

        # Check if any results were returned
        if not results:
            return jsonify(success=False, message="No keypoints detected"), 404

        # Extract the predictions (keypoints)
        result = results[0]  # Assuming one image
        keypoints = result.keypoints.xy[0].cpu().numpy()  # Extract keypoints for the detected object

        # Prepare the figure for visualization
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(original_image)
        ax.axis('off')

        heights = []  # Store heights for each vertebra
        classifications = []  # Store classification results for each vertebra
        fractured_vertebrae = []  # List to store fractured vertebrae names
        num_fractures = 0  # Count the number of compression fractures

        num_vertebrae = 5  # We're processing L1-L5, so 5 vertebrae
        for i in range(num_vertebrae):
            # Get the 4 keypoints for the current vertebra (i-th vertebra)
            k1 = keypoints[i * 4 + 0]  # Keypoint 1 (left-top)
            k2 = keypoints[i * 4 + 1]  # Keypoint 2 (right-top)
            k3 = keypoints[i * 4 + 2]  # Keypoint 3 (left-bottom)
            k4 = keypoints[i * 4 + 3]  # Keypoint 4 (right-bottom)

            # Calculate the heights for each side of the vertebra
            left_height = euclidean_distance(k1, k3)  # Distance between keypoint 1 and 3
            right_height = euclidean_distance(k2, k4)  # Distance between keypoint 2 and 4
            heights.append((left_height, right_height))

            # Classify the height loss and check for compression fracture
            classification, height_loss_percentage = classify_height_loss(left_height, right_height)
            classifications.append((classification, height_loss_percentage))

            # Increment the fracture count if a fracture is detected and add vertebra name
            if classification != "No compression fracture":
                num_fractures += 1
                fractured_vertebrae.append(vertebrae_map[i])

            # Draw blue lines to represent the heights for each vertebra
            ax.plot([k1[0], k3[0]], [k1[1], k3[1]], 'b-', lw=2)  # Line between keypoint 1 and 3 (left side)
            ax.plot([k2[0], k4[0]], [k2[1], k4[1]], 'b-', lw=2)  # Line between keypoint 2 and 4 (right side)

            # Mark the keypoints for visualization (optional)
            ax.scatter([k1[0], k2[0], k3[0], k4[0]], [k1[1], k2[1], k3[1], k4[1]], c='red', s=40)

        # Save the image with keypoints and lines
        output_uuid = str(uuid.uuid4())
        output_image_path = os.path.join(app.config['IMAGE_FOLDER'], f"{output_uuid}_keypoints.png")
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Prepare the response with the number of fractures and list of fractured vertebrae
        return jsonify(
            success=True,
            imageUrl=f"/static/images/{output_uuid}_keypoints.png",
            numFractures=num_fractures,  # Number of detected compression fractures
            fracturedVertebrae=fractured_vertebrae  # List of fractured vertebrae
        )

    except Exception as e:
        print(str(e))
        return jsonify(success=False, message=str(e)), 500

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
        midsagittal_slice_rotated = midsagittal_slice
        mask_sagittal_slice_rotated = mask_sagittal_slice

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

from shapely.geometry import LineString, Point

@app.route('/calculate-spondylolisthesis', methods=['GET'])
def calculate_spondylolisthesis():
    id = request.args.get('id')

    if not id:
        return jsonify(success=False, message="Missing parameters"), 400

    # Load the uploaded image
    image_filenames = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.startswith(id)]
    if not image_filenames:
        return jsonify(success=False, message="Image not found"), 404

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filenames[0])
    original_image = Image.open(image_path)

    # Perform inference to get the predicted keypoints
    results = xr_model.predict(source=image_path, save=False)

    # Check if any results were returned
    if not results:
        return jsonify(success=False, message="No keypoints detected"), 404

    # Extract the predictions (keypoints)
    result = results[0]  # Assuming one image
    keypoints = result.keypoints.xy[0].cpu().numpy()  # Extract keypoints for the detected object

    # Ensure that we have enough keypoints
    expected_keypoints = 22  # 5 vertebrae * 4 keypoints + 2 keypoints for S1
    if keypoints.shape[0] < expected_keypoints:
        return jsonify(success=False, message="Couldn't identify all vertebrae on x-ray"), 400

    # Indexing for L5 and S1 keypoints
    # L5 keypoints indices: 16 to 19
    # S1 keypoints indices: 20 and 21

    # Keypoints for L5 (posterior cortex)
    k2_L5 = keypoints[16 + 1]  # Second keypoint of L5 (right superior corner)
    k4_L5 = keypoints[16 + 3]  # Fourth keypoint of L5 (right inferior corner)

    # Line representing posterior cortex of L5
    posterior_cortex_L5 = [k2_L5, k4_L5]

    # Keypoints for S1 superior endplate
    k1_S1 = keypoints[20]  # Left superior corner of S1
    k2_S1 = keypoints[21]  # Right superior corner of S1

    # Line representing superior endplate of S1
    superior_endplate_S1 = [k1_S1, k2_S1]

    # Divide the S1 superior endplate into four equal parts
    # Calculate points at 25%, 50%, and 75% along the line
    def interpolate_points(p1, p2, fractions):
        points = []
        for f in fractions:
            x = p1[0] + f * (p2[0] - p1[0])
            y = p1[1] + f * (p2[1] - p1[1])
            points.append([x, y])
        return points

    fractions = [0.25, 0.5, 0.75]
    division_points = interpolate_points(k1_S1, k2_S1, fractions)

    # Prepare the figure for visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(original_image)
    ax.axis('off')

    # Plot the S1 superior endplate
    ax.plot([k1_S1[0], k2_S1[0]], [k1_S1[1], k2_S1[1]], 'g-', lw=2, label='S1 Superior Endplate')

    # Plot the division lines
    quadrant_lines = []
    for idx, point in enumerate(division_points):
        # Calculate orthogonal lines at division points
        def orthogonal_line(p1, p2, point):
            # Get direction vector of the original line
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            # Orthogonal direction
            ortho_dx = -dy
            ortho_dy = dx
            # Normalize
            length = np.hypot(ortho_dx, ortho_dy)
            ortho_dx /= length
            ortho_dy /= length
            # Extend the line for plotting
            factor = 1000  # Adjust as needed
            x1 = point[0] - ortho_dx * factor
            y1 = point[1] - ortho_dy * factor
            x2 = point[0] + ortho_dx * factor
            y2 = point[1] + ortho_dy * factor
            return [x1, x2], [y1, y2]

        x_line, y_line = orthogonal_line(k1_S1, k2_S1, point)
        ax.plot(x_line, y_line, 'm--', lw=1)
        quadrant_lines.append(LineString([(x_line[0], y_line[0]), (x_line[1], y_line[1])]))

    # Plot the posterior cortex of L5
    ax.plot([k2_L5[0], k4_L5[0]], [k2_L5[1], k4_L5[1]], 'r-', lw=2, label='L5 Posterior Cortex')

    # Create LineString objects
    line_posterior_cortex = LineString([k2_L5, k4_L5])
    line_superior_endplate = LineString([k1_S1, k2_S1])

    # Find intersection point between posterior cortex of L5 and S1 superior endplate
    intersection = line_posterior_cortex.intersection(line_superior_endplate)

    if intersection.is_empty or not isinstance(intersection, Point):
        # If no intersection, consider no slippage
        classification = "No anterolisthesis"
        slippage_percentage = 0.0
        # For visualization, project the posterior cortex onto the S1 endplate
        point_to_project = Point(line_posterior_cortex.coords[0])  # Convert to Point object
        projected_distance = line_superior_endplate.project(point_to_project)
        projected_point = line_superior_endplate.interpolate(projected_distance)
        intersection_x = projected_point.x
        intersection_y = projected_point.y
        # Plot the projected point
        ax.plot(intersection_x, intersection_y, 'bo', markersize=8, label='Projected Point')
    else:
        # Project the intersection point onto the S1 superior endplate
        total_length = line_superior_endplate.length
        distance_along_line = line_superior_endplate.project(intersection)
        proportion = distance_along_line / total_length

        # Determine the quadrant based on the proportion
        if proportion <= 0.25:
            classification = "Grade I"
        elif proportion <= 0.5:
            classification = "Grade II"
        elif proportion <= 0.75:
            classification = "Grade III"
        elif proportion <= 1.0:
            classification = "Grade IV"
        else:
            classification = "Grade V (Spondyloptosis)"

        slippage_percentage = proportion * 100

        # Plot the intersection point
        ax.plot(intersection.x, intersection.y, 'bo', markersize=8, label='Intersection Point')

    # Add legends and annotations
    ax.legend()

    # Save the image with lines
    output_uuid = str(uuid.uuid4())
    output_image_path = os.path.join(app.config['IMAGE_FOLDER'], f"{output_uuid}_spondylolisthesis.png")
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Prepare the response
    return jsonify(
        success=True,
        imageUrl=f"/static/images/{output_uuid}_spondylolisthesis.png",
        slippagePercentage=f"{slippage_percentage:.2f}%",
        meyerdingClassification=classification
    )

@app.route('/calculate-insurance', methods=['GET', 'POST'])
def calculate_insurance():
    # If it's a POST request, handle the CSV file upload
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'Missing required CSV file', 400

        uploaded_file = request.files['file']
        
        if uploaded_file.filename == '':
            return 'No selected file', 400
        
        # Read the uploaded CSV file
        csv_file = io.StringIO(uploaded_file.stream.read().decode('UTF-8'))
        
    # If it's a GET request, handle the radiology report text
    elif request.method == 'GET':
        radiology_report = request.args.get('radiology-report')

        if not radiology_report:
            return 'Missing required parameter: radiology-report', 400

        # Create a CSV file from the radiology report data
        csv_file = io.StringIO()
        writer = csv.DictWriter(csv_file, fieldnames=[
            'institution', 'PatientID', 'AccessionNumber',
            'deid_english_text', 'PatientSex', 'PatientAge', 'StudyDate'
        ])
        writer.writeheader()
        writer.writerow({
            'institution': '',
            'PatientID': '',
            'AccessionNumber': '',
            'deid_english_text': radiology_report,
            'PatientSex': "",
            'PatientAge': "",
            'StudyDate': ''
        })
        csv_file.seek(0)

    # Send POST request to the external service with the CSV file
    files = {'filename': ('input.csv', csv_file.read(), 'text/csv')}  # 'filename' key points to the file tuple

    response = requests.post(
        'https://www.nlp.neuro-innovate.com/neuroinnov/multiple_files',
        files=files  # Send the file with the correct 'filename' key
    )

    if response.status_code == 200:
        # Process the response
        response_csv = io.StringIO(response.text)
        reader = csv.DictReader(response_csv)
        results = list(reader)
        result_data = results[0]  # Assuming only one record

        # Return JSON response with the result
        return jsonify(result_data)
    else:
        return 'Error in processing the insurance calculation.', 500
@app.route('/output_xr')
def output_xr():
    return render_template('output_xr.html')

@app.route('/output_ct')
def output_ct():
    return render_template('output_ct.html')

@app.route('/output_mri')
def output_mri():
    return render_template('output_mr.html')
@app.route('/output_insurance')
def output_insurance():
    return render_template('output_insurance.html')

@app.route('/loading')
def loading():
    return render_template('loading.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=sys.argv[1], debug=True)
