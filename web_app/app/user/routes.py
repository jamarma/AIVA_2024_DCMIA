import numpy as np
import cv2
import os
from flask import render_template, request, current_app, redirect, url_for, send_from_directory
from flask_login import login_required, current_user

from . import user_bp
from . import forms
from dcmia.house_detector import HouseDetector
from dcmia import utils


@user_bp.route('/process_image', methods=['GET', 'POST'])
@login_required
def process_image():
    """
    Renders the page with the form to upload an image and runs
    the house detection algorithm. Also saves the results
    of detections in media folder.
    """
    upload_form = forms.UploadForm(request.form)
    if request.method == 'POST':
        # Reads image and converts it to numpy array
        file = request.files['image'].read()
        image = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)

        # House detection
        detector = HouseDetector(model_filename='model1_fcos5.pth')
        boxes, labels, scores = detector.detect(image)
        output = utils.draw_boxes(image, boxes)

        # Saves image in png format to media dir
        media_dir = os.path.join(current_app.config['MEDIA_DIR'], current_user.username)
        os.makedirs(media_dir, exist_ok=True)
        cv2.imwrite(os.path.join(media_dir, 'result.png'), output)

        # Saves predictions in txt file to media dir
        utils.save_predictions(boxes, os.path.join(media_dir, 'result.txt'))
        return redirect(url_for('user.results'))
    return render_template("user/process_image.html", form=upload_form)


@user_bp.route('/results', methods=['GET', 'POST'])
@login_required
def results():
    """Renders the page with the results of house detection."""
    image_file = os.path.join(current_user.username, 'result.png')
    results_file = os.path.join(current_user.username, 'result.txt')
    results_file_path = os.path.join(current_app.config['MEDIA_DIR'], results_file)
    with open(results_file_path, 'r') as file:
        num_houses = file.readline().split(':')[1]
    return render_template("user/results.html",
                           image_file=image_file,
                           results_file=results_file,
                           num_houses=num_houses)


@user_bp.route('/media/<path:filename>')
@login_required
def media_results(filename):
    """Serves a file from media dir."""
    dir_path = current_app.config['MEDIA_DIR']
    return send_from_directory(dir_path, filename)


@user_bp.route('/media/<path:filename>')
def download_file(filename):
    """Serves a file from media dir to download."""
    dir_path = current_app.config['MEDIA_DIR']
    return send_from_directory(dir_path, filename, as_attachment=True)



