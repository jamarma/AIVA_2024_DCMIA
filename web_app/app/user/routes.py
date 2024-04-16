import numpy as np
import cv2
import time
from flask import render_template, request
from flask_login import login_required

from . import user_bp
from . import forms


@user_bp.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    upload_form = forms.UploadForm(request.form)
    if request.method == 'POST':
        file = request.files['image'].read()
        image = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)
        print(f'Reads image with shape {image.shape}')
        for i in range(5):
            time.sleep(1)
    return render_template("user/dashboard.html", form=upload_form)

