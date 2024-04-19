from flask import render_template, redirect, url_for, request, flash
from flask_login import login_user, login_required, logout_user, current_user

from .models import User
from . import auth_bp
from . import forms
from app import login_manager


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Handles user login and renders the login page."""
    if current_user.is_authenticated:
        return redirect(url_for('user.process_image'))
    login_form = forms.LoginForm(request.form)
    if request.method == 'POST' and login_form.validate():
        user = User.get_by_username(login_form.username.data)
        if not user or not user.verify_password(login_form.password.data):
            flash('Please check your login details and try again.', "danger")
            return redirect(url_for('auth.login'))
        login_user(user, remember=login_form.remember_me.data)
        return redirect(url_for('user.process_image'))
    return render_template("auth/login.html", form=login_form)


@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    """Handles user signup and renders the signup page."""
    if current_user.is_authenticated:
        return redirect(url_for('user.process_image'))
    signup_form = forms.SignupForm(request.form)
    if request.method == 'POST' and signup_form.validate():
        user = User(first_name=signup_form.first_name.data,
                    last_name=signup_form.last_name.data,
                    email=signup_form.email.data,
                    username=signup_form.username.data,
                    password=signup_form.password.data)
        user.save()
        flash("Successfully registered!", "success")
        return redirect(url_for('auth.login'))
    return render_template("auth/signup.html", form=signup_form)


@auth_bp.route('/logout')
@login_required
def logout():
    """Handles user logout and redirects to login page."""
    logout_user()
    return redirect(url_for('auth.login'))


@auth_bp.route('/')
def index():
    """Redirects to login page."""
    return redirect(url_for('auth.login'))


@login_manager.user_loader
def load_user(user_id: int):
    """
    Loads a user from database and returns it.

    Parameters:
        - user_id (int): the id of the user.

    Returns:
        - User: the user with given id.
    """
    return User.get_by_id(int(user_id))
