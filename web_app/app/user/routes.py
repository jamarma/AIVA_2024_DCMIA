from flask import render_template
from flask_login import login_required

from . import user_bp


@user_bp.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    return render_template("user/dashboard.html")
