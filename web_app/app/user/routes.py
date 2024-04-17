import numpy as np
import cv2
import os
from flask import render_template, request, current_app, redirect, url_for, send_from_directory
from flask_login import login_required, current_user

from . import user_bp
from . import forms
from .dcmia.src.house_detector import HouseDetector
from .dcmia.src import utils


@user_bp.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    upload_form = forms.UploadForm(request.form)
    if request.method == 'POST':
        file = request.files['image'].read()
        image = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)

        # Detect houses
        detector = HouseDetector(model_filename='model1_fcos5.pth')
        boxes, labels, scores = detector.detect(image)
        output = utils.draw_boxes(image, boxes)

        # Save image to visualize in browser
        images_dir = os.path.join(current_app.config['MEDIA_DIR'], current_user.username)
        os.makedirs(images_dir, exist_ok=True)
        cv2.imwrite(os.path.join(images_dir, 'result.png'), output)
        return redirect(url_for('user.results'))
    return render_template("user/dashboard.html", form=upload_form)


@user_bp.route('/results', methods=['GET', 'POST'])
@login_required
def results():
    image = os.path.join(current_user.username, 'result.png')
    return render_template("user/results.html", image=image)


@user_bp.route('/media/<path:filename>')
@login_required
def media_results(filename):
    dir_path = current_app.config['MEDIA_DIR']
    return send_from_directory(dir_path, filename)


@user_bp.route('/media/<path:filename>')
def download_file(filename):
    dir_path = current_app.config['MEDIA_DIR']
    return send_from_directory(dir_path, filename, as_attachment=True)
