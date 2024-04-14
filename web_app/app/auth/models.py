from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

from app.db import db, BaseModelMixin


class User(UserMixin, db.Model, BaseModelMixin):
    """Class that represents a user of app in the SQL database."""
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(15), nullable=False)
    last_name = db.Column(db.String(15), nullable=False)
    email = db.Column(db.String(40), unique=True, nullable=False)
    username = db.Column(db.String(15), unique=True, nullable=False)
    password = db.Column(db.String(102), nullable=False)

    def __init__(self, first_name: str, last_name: str, email: str, username: str, password: str):
        """
        Initializes a User.

        Parameters:
            - first_name (str).
            - last_name (str).
            - email (str).
            - username (str).
            - password (str).
        """
        self.first_name = first_name
        self.last_name = last_name
        self.email = email
        self.username = username
        self.password = generate_password_hash(password)

    def verify_password(self, password: str):
        """
        Verifies that the given password matches with the user password.

        Parameters:
            - password (str): the given password.

        Returns:
            - bool: True if the passwords match, False otherwise.
        """
        return check_password_hash(self.password, password)

    def update_password(self, password: str):
        """
        Updates the user password in the database.

        Parameters:
             - password (str): the new password.
        """
        self.password = generate_password_hash(password)

    @staticmethod
    def get_by_username(username: str):
        """
        Takes a user from database that has the given username.

        Parameters:
            - username (str).

        Returns:
            - User: the user that has the given username.
        """
        return User.query.filter_by(username=username).first()

    @staticmethod
    def get_by_email(email):
        """
        Takes a user from database that has the given email.

        Parameters:
            - email (str).

        Returns:
            - User: the user that has the given email.
        """
        return User.query.filter_by(email=email).first()
