import flask
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import io
import base64
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_edges(image_data, blur_size=5, threshold1=50, threshold2=150):
    """Apply edge detection to an image"""
    try:
        # Convert string of image data to uint8
        nparr = np.frombuffer(image_data, np.uint8)
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        # Apply Gaussian Blur
        blurred_image = cv2.GaussianBlur(image, (blur_size, blur_size), 1.4)

        # Apply Canny edge detector
        edges = cv2.Canny(blurred_image, threshold1, threshold2)

        return edges, None

    except Exception as e:
        return None, str(e)


@app.route('/', methods=['GET', 'POST'])
def index():
    original_image = None
    edges_image = None
    error = None

    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)

        file = request.files['file']

        # Check if file is selected
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)

        # Check if file is allowed
        if file and allowed_file(file.filename):
            # Read file data
            file_data = file.read()

            # Get parameters from form
            try:
                blur_size = int(request.form.get('blur_size', 5))
                # Ensure blur size is odd
                if blur_size % 2 == 0:
                    blur_size += 1
                threshold1 = int(request.form.get('threshold1', 50))
                threshold2 = int(request.form.get('threshold2', 150))
            except ValueError:
                flash('Invalid parameter values')
                return redirect(request.url)

            # Detect edges
            edges, error = detect_edges(file_data, blur_size, threshold1, threshold2)

            if error:
                flash(f'Error processing image: {error}')
                return redirect(request.url)

            # Convert images to base64 for displaying in HTML
            # Original image
            nparr = np.frombuffer(file_data, np.uint8)
            original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            _, original_buffer = cv2.imencode('.png', original_img)
            original_image = base64.b64encode(original_buffer).decode('utf-8')

            # Edge detected image
            _, edges_buffer = cv2.imencode('.png', edges)
            edges_image = base64.b64encode(edges_buffer).decode('utf-8')

        else:
            flash('Invalid file type. Please upload an image file (PNG, JPG, JPEG, BMP, TIFF).')
            return redirect(request.url)

    return render_template('index.html',
                           original_image=original_image,
                           edges_image=edges_image,
                           error=error)


if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    app.run(debug=True, host='0.0.0.0', port=5000)