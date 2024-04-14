from wtforms import Form
from wtforms import StringField, BooleanField, PasswordField
from wtforms import validators
from .models import User


class LoginForm(Form):
    """Represents a web login form."""
    username = StringField('Username', [
        validators.DataRequired(),
        validators.Length(min=4, max=15)
    ])
    password = PasswordField('Password', [
        validators.DataRequired()
    ])
    remember_me = BooleanField('Keep me logged in')


class SignupForm(Form):
    """Represents a web signup form."""
    first_name = StringField('First Name', [
        validators.DataRequired(),
        validators.Length(min=3, max=15)
    ])
    last_name = StringField('Last Name', [
        validators.DataRequired(),
        validators.Length(min=3, max=15)
    ])
    username = StringField('Username', [
        validators.DataRequired(),
        validators.Length(min=4, max=15)
    ])
    email = StringField('Email', [
        validators.DataRequired(),
        validators.Email(),
        validators.Length(max=40)
    ])
    password = PasswordField('Password', [
        validators.DataRequired()
    ])
    confirm = PasswordField('Repeat password', [
        validators.DataRequired(),
        validators.EqualTo('password', message='Passwords must match')
    ])

    def validate_username(form, field):
        """Checks if the username already exists in the database."""
        username = field.data
        user = User.get_by_username(username)
        if user is not None:
            raise validators.ValidationError('Username already exists')

    def validate_email(form, field):
        """Checks if the email already exists in the database."""
        email = field.data
        user = User.get_by_email(email)
        if user is not None:
            raise validators.ValidationError('Email already exists')
